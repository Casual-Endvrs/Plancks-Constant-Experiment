#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:14:07 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import os
import sys
import serial.tools.list_ports

import pandas as pd
import numpy as np
import lmfit
import time

import random
import pickle

import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def waveLengthToRGB(Wavelength) :
    """
    This is based on the source: https://code.i-harness.com/en/q/167802
    This function returns the corresponding RGB code for a given wavelength,
        specified in nm, that is within the visible range, 380 nm to 780. 
        If the wavelength is not visible, the RGB code for black is returned.

    Parameters
    ----------
    Wavelength : float or int
        This is the wavelength of the light, with units nm, that is to be 
        converted to an RGB color code.

    Returns
    -------
    Red : float
        Red component.
    Green : float
        Green component.
    Blue : float
        Blue component.
    """
    if Wavelength >= 380 and Wavelength<440 :
        Red = -(Wavelength - 440) / (440 - 380);
        Green = 0;
        Blue = 1;
    elif Wavelength >= 440 and Wavelength<490 :
        Red = 0;
        Green = (Wavelength - 440) / (490 - 440);
        Blue = 1;
    elif Wavelength >= 490 and Wavelength<510 :
        Red = 0;
        Green = 1;
        Blue = -(Wavelength - 510) / (510 - 490);
    elif Wavelength >= 510 and Wavelength<580 :
        Red = (Wavelength - 510) / (580 - 510);
        Green = 1;
        Blue = 0;
    elif Wavelength >= 580 and Wavelength<645 :
        Red = 1;
        Green = -(Wavelength - 645) / (645 - 580);
        Blue = 0;
    elif Wavelength >= 645 and Wavelength<781 :
        Red = 1;
        Green = 0;
        Blue = 0;
    else :
        Red = 0;
        Green = 0;
        Blue = 0;
    
    return (Red, Green, Blue)

def linear_fn(x, m, b) :
    """
    This function returns the simple equation: y=mx+b.

    Parameters
    ----------
    x : float
        Independent variable.
    m : float
        Defines the graphs slope.
    b : float
        Defines the graphs y-intercept.

    Returns
    -------
    float
        The result of m*x+b.

    """
    return m*x+b

class Planck_experiment() :
    """
    This class is used to store all pertinant information about the Arduino
        and experimental results.
    
    Attributes
    ----------
    port_str : str
        Stores the port string that is required to connect to the Arduino.
    
    baud : int
        Bits per second communication rate to be used with the Arduino. This
        value must be the same as that used in the Arduino sketch.
    
    arduino : pySerial object
        Sorts the pySerial object that is attributed to the Arduino. All serial 
        commands to the Arduino are sent via this object.
    
    colors : list[ str ]
        This is a list of strings containing the color label for each LED that 
        is used for the experiment.
    
    wavelengths : dictionary[ str:int ]
        Dictionary used to convert an LED's color label to that LED's wavelength.
    
    colors_known_wavelength : list[ int ]
        A list storing the wavelength
    
    wavelengths_analysis
    
    dfs
    
    analysis_results
    
    knee_voltages
    
    clr_df_updated
    
    save_folder
    
    current_data
    
    calcd_planck_constant
    
    plancks_fit_results
    
    Methods
    -------
    
    """
    def __init__(self) :
        """
        Constructs the Planck_experiment class.

        Returns
        -------
        None.

        """
        self.port_str = ""
        self.baud = 9600
        self.arduino = None
        
        self.colors = []
        self.wavelengths = {}
        self.colors_known_wavelength = []
        self.wavelengths_analysis = []
        self.dfs = {}
        self.analysis_results = {}
        self.knee_voltages = []
        
        # This term keeps track if the LED color was last: 
        #   True) updated with new data, 
        #   False) had experimental results processing.
        self.clr_df_updated = {}
        
        self.save_folder = None
        
        self.current_data = []
        self.calcd_planck_constant = None
        self.plancks_fit_results = None
    
    def connect_to_arduino(self, port_str) :
        """
        Connects to an Arduino on port port_str.

        Parameters
        ----------
        port_str : string
            Port address for the Arduino.

        Returns
        -------
        str
            Description of connection result.

        """
        if self.arduino is not None :
            self.arduino.close()
        
        self.port_str = port_str
        
        try :
            self.arduino = serial.Serial(self.port_str, self.baud)
            time.sleep(0.25)
            return 'Success'
        except :
            return 'Connection Failure'
    
    def disconnect_from_arduino(self) :
        """
        Disconnects from the Arduino.

        Returns
        -------
        None.

        """
        if self.arduino is not None :
            self.arduino.close()
    
    def add_exp_data(self, color) :
        """
        Saves experimental result to data frame.

        Parameters
        ----------
        color : str
            Name of the color of LED used for experiment.

        Returns
        -------
        None.

        """
        if color in self.colors :
            df = self.dfs[ color ]
            col_name = 'LED %i [V]' %len( df.columns )
            df[ col_name ] = self.current_data
            self.clr_df_updated[color] = True
    
    def add_color(self, color, nm) :
        """
        Adds an LED color to the data frame.

        Parameters
        ----------
        color : str
            Color of the LED.
        nm : str
            Wavelength or wavelength range fo the LED.

        Returns
        -------
        None.

        """
        self.colors.append( color )
        self.wavelengths[color] = nm
        self.dfs[color] = pd.DataFrame()
        
        self.clr_df_updated[color] = True
    
    def turn_on_Vcc_correction(self) :
        """
        Send command to Arduino to enable Vcc correction.

        Returns
        -------
        None.

        """
        if self.arduino is not None :
            self.arduino.write( b'CVcc,t/' )
    
    def turn_off_Vcc_correction(self) :
        """
        Send command to Arduino to disable Vcc correction.
        
        Returns
        -------
        None.

        """
        if self.arduino is not None :
            self.arduino.write( b'CVcc,f/' )
    
    def dump_dfs(self) :
        """
        Saves data frame of experimental results to file as pickle.

        Returns
        -------
        None.

        """
        if self.save_folder is not None :
            with open(os.path.join(self.save_folder, 'LED_wavelengths.pkl'), 'wb') as f :
                pickle.dump(self.wavelengths, f)
            
            for clr in self.colors:
                fil = '%s-dataFrame.pkl' %clr
                with open(os.path.join(self.save_folder, fil), 'wb') as f:
                    pickle.dump(self.dfs[clr], f)

    def save_dfs(self) :
        """
        Saves data frame of experimental results to csv file.

        Returns
        -------
        None.

        """
        if self.save_folder is not None :
            for clr in self.colors :
                df = self.dfs[clr]
                # inserting a column for input voltages
                if 'Input Voltage [V]' not in df.columns :
                    df.insert(0, 'Input Voltage [V]', pd.Series( np.arange(4096)/4095*5.0 ))
                fil = '%s.csv' %clr
                with open(os.path.join(self.save_folder, fil), 'w') as f:
                    f.write( '# These are the results for LEDs of the color: %s\r\n' %clr )
                    f.write( '# This color has a wavelength in the range of: %s nm\r\n' %self.wavelengths[clr] )
                    f.write( '# All values in the table are voltages with units Volts.\r\n' )
                    f.write( '# stdom is short for standard deviation of the mean.\r\n' )
                    f.write( '# Input voltage has no error measurements. Improvements should be made.\r\n' )
                    f.write( '\r\n' )
                    df.to_csv( f )
                    f.write( '\r\n\r\n\r\n' )
    
    def load_dfs(self, progress_bar=None) :
        """
        Loads data frame of experimental results from pickle files.

        Parameters
        ----------
        progress_bar : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if self.save_folder is not None :
            pkl_fils = []
            
            use_prog_bar = False
            if progress_bar is not None :
                use_prog_bar = True
                progress_bar.setValue(0)
            
            for root, dir, fils in os.walk(self.save_folder) :
                for f in fils :
                    if ".pkl" in f and f != 'LED_wavelengths.pkl' :
                        pkl_fils.append( f )
            
            num_fils = len(pkl_fils)
            
            for fil_num in np.arange(num_fils) :
                if use_prog_bar :
                    progress_bar.setValue( 100*fil_num/(num_fils+1) )
                
                fil = pkl_fils[fil_num]
                with open(os.path.join(self.save_folder, fil), 'rb') as pkl_data :
                    clr = fil.split('-')[0]
                    self.dfs[clr] = pickle.load( pkl_data )
                    self.colors.append( clr )
                    self.clr_df_updated[clr] = True
            
            if use_prog_bar :
                progress_bar.setValue( 100*(num_fils)/(num_fils+1) )
            
            with open(os.path.join(self.save_folder, 'LED_wavelengths.pkl'), 'rb') as f :
                self.wavelengths = pickle.load(f)
            
            if use_prog_bar :
                progress_bar.setValue( 100 )
    
    def process_results(self, color=None, progress_bar=None) :
        """
        Calculates the knee voltage for the LED of "color".

        Parameters
        ----------
        color : str, optional
            Color of the LED. If None, all LED results will be calculated.
            The default is None.
        progress_bar : TYPE, PyQT5 QProgressBar
            Progress bar used to show progression through analysis.
            The default is None.

        Returns
        -------
        None.

        """
        use_prog_bar = False
        if progress_bar is not None :
            use_prog_bar = True
            progress_bar.setValue( 0 )
        
        if color is None :
            colors = self.colors
        else :
            colors = [color]
        
        num_clrs = len(colors)
        
        for idx in np.arange( num_clrs ) :
            if use_prog_bar :
                progress_bar.setValue( 100*idx/num_clrs )
            clr = colors[idx]
            if not self.clr_df_updated[clr] :
                continue
            self.clr_df_updated[clr] = False
            
            if clr not in self.colors :
                continue
            df = self.dfs[ clr ]
            if not df.empty :
                if not df.empty :
                    mns = []
                    stdom = []
                    sqrt_num_data = np.sqrt( len( df.loc[ 0 ] ) )
                    
                    cols = [ col for col in df.columns if 'LED' in col ]
                    selected_data = df[cols]
                    
                    for i in df.index :
                        data = selected_data.loc[ i ]
                        mns.append( np.mean(data) )
                        stdom.append( np.std(data)/sqrt_num_data )
                    df[ 'mean [V]' ] = mns
                    df[ 'stdom [V]' ] = stdom
                    
                    data = df['mean [V]'].to_numpy()
                    mx_val = data[-1]
                    
                    low = data>0.1
                    high = data<(0.98*mx_val)
                    selected = low&high
                    
                    V_input = np.arange(4096)*5.0/4095
                    model = lmfit.Model( linear_fn )
                    params = lmfit.Parameters()
                    params.add('m', value=1, vary=True)
                    params.add('b', value=-1, vary=True)
                    result_high = model.fit(data[selected], params, x=V_input[selected])
                    
                    knee_voltage = -1 * result_high.values['b'] / result_high.values['m']
                    self.analysis_results[clr] = [knee_voltage, 
                                                  result_high.values['m'], 
                                                  result_high.values['b'], 
                                                   result_high.best_fit]
        if use_prog_bar :
            progress_bar.setValue( 100 )
    
    def calc_Plancks_constant(self) :
        """
        Calculates Planck's constant from all LED knee voltage results.

        Returns
        -------
        None.

        """
        self.knee_voltages = []
        self.wavelengths_analysis = []
        self.colors_known_wavelength = []
        
        num_data_sets = len(self.wavelengths)
        
        for idx in np.arange( num_data_sets ) :
            clr = self.colors[idx]
            wl = self.wavelengths[clr].split('-')
            
            if wl[0] == '???' or clr not in self.analysis_results :
                continue
            if len(wl)==2 :
                wl = ( int(wl[0]) + int(wl[1]) ) / 2.
            else :
                wl = int(wl[0])
            
            self.knee_voltages.append( self.analysis_results[clr][0] )
            self.colors_known_wavelength.append( clr )
            self.wavelengths_analysis.append( wl )
        
        self.knee_voltages = np.array( self.knee_voltages )
        self.wavelengths_analysis = np.array( self.wavelengths_analysis )
        
        if len( self.knee_voltages ) > 1 :
            c_over_e = 3e8 / 1.602e-19
            
            model = lmfit.Model( linear_fn )
            params = lmfit.Parameters()
            params.add('m', value=0, vary=True)
            params.add('b', value=0, vary=True)
            results = model.fit(self.knee_voltages, params, x=1e9/self.wavelengths_analysis)
            self.plancks_fit_results = results
            
            self.calcd_planck_constant = results.values['m']/c_over_e
    
    def plot_knee_voltages(self, canvas) :
        """
        Plots the results to show knee voltage for each LED.

        Parameters
        ----------
        canvas : MplCanvas object
            Canvas to plot results on.

        Returns
        -------
        None.

        """
        if self.calcd_planck_constant is not None and self.plancks_fit_results is not None :
            
            fit_result = self.plancks_fit_results.best_fit
            self.wavelengths_analysis, self.colors_known_wavelength, self.knee_voltages, fit_result = \
                zip(*sorted(zip(self.wavelengths_analysis, self.colors_known_wavelength, self.knee_voltages, fit_result), reverse=False))
            
            self.wavelengths_analysis = np.array(self.wavelengths_analysis)
            x = 1e9/self.wavelengths_analysis
            
            canvas.axes.cla()
            canvas.axes.plot(x, fit_result)
            for idx in np.arange( len(self.wavelengths_analysis) ) :
                clr = waveLengthToRGB(self.wavelengths_analysis[idx])
                canvas.axes.plot(x[idx], self.knee_voltages[idx], 'o', color='k', markersize=7)
                canvas.axes.plot(x[idx], self.knee_voltages[idx], 'o', color=clr, markersize=5, 
                                 label=self.colors_known_wavelength[idx])
            
            
            fontsize = 15
            error = 100 * ( self.calcd_planck_constant - 6.62607015e-34 ) / self.calcd_planck_constant
            canvas.axes.text(np.min(x), 0.95*np.max(self.knee_voltages), 
                "Experimental Result for\nPlanck\'s Constant: %.2e Js" %self.calcd_planck_constant, 
                fontsize=fontsize)
            canvas.axes.text(np.min(x), 0.9*np.max(self.knee_voltages), 
                "Error: %.2f%%" %error, fontsize=fontsize)
            
            canvas.axes.set_xlabel("Inverse of Wavelength [1e6 1/nm]")
            canvas.axes.set_ylabel("Knee Voltage [V]")
            
            x_labels = [ '%.1f' %(1e-6*i) for i in canvas.axes.get_xticks() ]
            canvas.axes.set_xticklabels(x_labels, fontsize=fontsize)
            canvas.axes.set_yticklabels(canvas.axes.get_yticks(), fontsize=fontsize)
            canvas.axes.legend(loc=4, prop={'size': fontsize})
            canvas.draw()
    
    def create_LED_result_plot(self, canvas, color) :
        """
        Plots the experimental result of an LED with an unspecified wavelength
        along with its calculated wavelength.

        Parameters
        ----------
        canvas : MplCanvas object
            Canvas to plot results on.
        color : str
            Color of LED to plot results for.

        Returns
        -------
        None.

        """
        if color in self.colors :
            df = self.dfs[color]
            if not df.empty :
                V_input = np.arange(4096)*5.0/4095
                data = df['mean [V]'].to_numpy()
                mx_val = data[-1]
                
                low = data>0.1
                high = data<(0.98*mx_val)
                selected = low&high
                
                canvas.axes.cla()
                canvas.axes.plot( V_input, data, '.-', label='Raw Data' )
                canvas.axes.plot( V_input[selected], self.analysis_results[color][3], label='Fit' )
                
                fontsize = 15
                height = 0.95
                canvas.axes.text(0.1, height*mx_val, 'LED Color: %s' %color, 
                                 fontsize=fontsize); height -= 0.075
                if self.wavelengths[color] == '???' :
                    if self.calcd_planck_constant is not None :
                        h = self.calcd_planck_constant
                        c_over_e = 3e8 / 1.602e-19
                        # c = 3e8
                        # e = 1.602e-19
                        V = self.analysis_results[color][0]
                        calc_wavelength = h * c_over_e / V
                        
                        canvas.axes.text(0.1, height*mx_val, 'Calculated Wavelength: %.0f nm' %(1e9*calc_wavelength), 
                                         fontsize=fontsize); height -= 0.075
                    else :
                        canvas.axes.text(0.1, height*mx_val, 'Calculated Wavelength: Unknown', 
                                         fontsize=fontsize); height -= 0.075
                else :
                    canvas.axes.text(0.1, height*mx_val, 'Wavelength [nm]: %s' %self.wavelengths[color], 
                                     fontsize=fontsize); height -= 0.075
                canvas.axes.text(0.1, height*mx_val, 'V_knee: %.3f [V]' %self.analysis_results[color][0], 
                                 fontsize=fontsize); height -= 0.075
                
                canvas.axes.set_xlabel('Supplied Voltage [V]', fontsize=fontsize)
                canvas.axes.set_ylabel('Voltage Across Resistor [V]', fontsize=fontsize)
                canvas.axes.set_xticklabels(canvas.axes.get_xticks(), fontsize=fontsize)
                canvas.axes.set_yticklabels(canvas.axes.get_yticks(), fontsize=fontsize)
                canvas.axes.legend(loc=4, prop={'size': fontsize})
                
                #canvas.fig.tight_layout()
                
                canvas.draw()
            else :
                canvas.axes.cla()
                canvas.axes.text(0.01, 0.5, 'There is no data corresponding to this LED color.', fontsize=fontsize)
                canvas.axes.xaxis.label.set_size(16)
                canvas.axes.yaxis.label.set_size(16)
                canvas.draw()
    
    def save_LED_figure_result(self, canvas, clr) :
        """
        Saves LED results to file.

        Parameters
        ----------
        canvas : MplCanvas object
            Canvas data was plotted on.
        clr : str
            Color of the LED.

        Returns
        -------
        None.

        """
        fil = clr+'-fit_result.png'
        canvas.fig.savefig(os.path.join(self.save_folder, fil), dpi=480)

class run_experiment(QThread) :
    """
    """
    notifyProgress = pyqtSignal(int)
    def __init__(self, experiment, canvas=None) :
        """
        Constructs the run_experiment class.

        Parameters
        ----------
        experiment : Planck_experiment object
            Stores experimental results.
        canvas : MplCanvas object, optional
            Canvas to plot results on. The default is None.

        Returns
        -------
        None.

        """
        QThread.__init__(self)
        self.experiment = experiment
        self.canvas = canvas
    
    def update_plot(self) :
        """
        Updates the experimental data in on screen plot.

        Returns
        -------
        None.

        """
        self.canvas.axes.cla()
        
        xdata = np.arange(len(self.experiment.current_data))*5.0/4095
        
        fontsize = 15
        self.canvas.axes.plot(xdata, self.experiment.current_data, 'r')
        self.canvas.axes.set_xlabel('Supplied Voltage [V]', fontsize=fontsize)
        self.canvas.axes.set_ylabel('Voltage Across Resistor [V]', fontsize=fontsize)
        self.canvas.axes.set_xticklabels(self.canvas.axes.get_xticks(), fontsize=fontsize)
        self.canvas.axes.set_yticklabels(self.canvas.axes.get_yticks(), fontsize=fontsize)
        
        last_val = self.experiment.current_data[-1]
        if last_val < 0.5 :
            str_fmt = "{x:.3f}"
        if last_val < 1 :
            str_fmt = "{x:.2f}"
        else :
            str_fmt = "{x:.1f}"
        self.canvas.axes.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        self.canvas.axes.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        self.canvas.fig.tight_layout()
        
        self.canvas.draw()
    
    def run(self) :
        """
        Runs experimental.

        Returns
        -------
        None.

        """
        self.experiment.current_data = []
        if self.experiment.arduino is None :
            return None
        
        self.experiment.arduino.write( b'run/' )
        itr = 0
        percent_last_update = 0
        while True :
            itr += 1
            result = self.experiment.arduino.readline()
            if result == b'end\r\n' :
                break
            self.experiment.current_data.append( float(result.strip()) )
            percent_complete = int(100*itr/4095)
            self.notifyProgress.emit( percent_complete )
            if self.canvas is not None and percent_complete >= percent_last_update :
                percent_last_update += 1
                self.update_plot()





def get_avail_ports() :
    """
    Search for all available ports with an Arduino connected.

    Returns
    -------
    avail_ports : list
        List of ports.

    """
    list_ports = serial.tools.list_ports.comports()
    
    avail_ports = []
    for port, desc, hwid in sorted(list_ports) :
        avail_ports.append( port )
    
    return avail_ports


class MainWindow(QMainWindow) :
    """
    """
    def __init__(self) :
        """
        Initialize MainWindow class.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.experiment = Planck_experiment()
        self.exp_connected = False
        self.added_new_color = False
        
        self.title = 'Planck\'s Constant Experiment'
        self.setWindowTitle(self.title)
        
        self.main_layout = QVBoxLayout()
        
        self.left = 0
        self.top = 0
        self.width = 900
        self.height = 600
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.intro_tab = intro_page(self)
        self.exp_controls = exp_controls(self)
        self.results_tab = results_tab(self)
        
        self.main_tabs = QTabWidget()
        
        self.main_tabs.addTab(self.intro_tab, "Introduction")
        self.main_tabs.addTab(self.exp_controls, "Experimental Controls")
        self.main_tabs.addTab(self.results_tab, "Experimental Analysis")
        
        self.main_tabs.currentChanged.connect(self.tab_changed)
        
        self.main_layout.addWidget(self.main_tabs)
        self.setCentralWidget(self.main_tabs)
        
        self.show()
    
    def tab_changed(self, idx) :
        """
        Function that runs upon tab change. It updates the color list if the
        new tab is the results_tab.

        Parameters
        ----------
        idx : int
            Index of the newly selected tab.

        Returns
        -------
        None.

        """
        if idx == 2 :
            self.results_tab.update_clr_list()
    
    def setup_experiment(self, port) :
        """
        Attempts to connect to the selected Arduino

        Parameters
        ----------
        port : str
            Port for the Arduino.

        Returns
        -------
        None.

        """
        result = self.experiment.connect_to_arduino(port)
        if result == 'Success' :
            self.exp_controls.enable_exp()
            self.exp_controls.update_voltage_correction()
        elif result == 'Connection Failure' :
            title = "Arduino Connection Failure"
            warning_msg = "\n".join(["An error occured while attempting to connect to the Arduino.", 
                "Please ensure that there are no other program connected to it;", 
                 "   for example, the Arduino IDE can cause connection issues.", 
                 "If no other programs are attempting to connect tothe Arduino,", 
                 "   press the reset button on your Arduino and then try again."])
            
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)

class intro_page(QWidget) :
    """
    Introduction page widget.
    Displays information about the experiment and allows the user to connect
        to a selected Arduino.
    """
    def __init__(self, parent) :
        """
        Constructs the intro_page class.

        Parameters
        ----------
        parent : MainWindow object
            MainWindow object that this object belongs to.

        Returns
        -------
        None.

        """
        super(QWidget, self).__init__(parent)
        max_widget_width = 150
        
        self.parent = parent
        self.avail_ports = [] # get_avail_ports()
        
        self.layout = QGridLayout(self)
        
        label_txt = "Select the port your Arduino is connected to: "
        label = QLabel(label_txt)
        label.setMaximumWidth(325)
        self.layout.addWidget(label, 0, 0)
        
        self.btn_check_devices = QPushButton("Check for Devices")
        self.btn_check_devices.setMaximumWidth(max_widget_width)
        self.btn_check_devices.clicked.connect(self.search_for_devices)
        self.layout.addWidget(self.btn_check_devices, 0, 1)
        
        self.ports_selection = QComboBox(self)
        self.layout.addWidget(self.ports_selection, 0, 2)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_exp)
        self.connect_btn.setMaximumWidth(max_widget_width)
        self.layout.addWidget(self.connect_btn, 0, 3)
        
        self.instruction_tabs = QTabWidget()
        
        self.intro_instructions = "\n".join([
            "Two websites that discuss the experiment and the theory and math behind it are:",
            "\thttps://educatech.in/measurement-of-plancks-constant-through-knee-voltage-of-leds/",
            "\thttps://www.scienceinschool.org/2014/issue28/planck",
            "",
            "Circuit pin connections: (Note: A circuit diagram is provided in the \"Circuit Diagram\" tab.)",
            "     MCP 4725 Vdd --> Aruidno +5V",
            "     MCP 4725 GND --> Aruidno GND",
            "     MCP 4725 SCL --> Aruidno A4",
            "     MCP 4725 SDA --> Aruidno A5",
            "     MCP 4725 A0 --> Aruidno GND",
            "     MCP 4725 Vout --> LED to be measured --> Resistor (2.2 kOhm) --> GND",
            "     (Optional): Arduino Pin 13 --> indicator LED --> Resistor (2.2 kOhm) --> GND",
            "",
            "-----------------------",
            "Data Aquisition:",
            "-----------------------",
            "If you are collecting data for the experiment using an Arduino then:",
            "     1) \"Check for devices\", select your Arduino and \"Connect\".",
            "     2) Navigate to \"Experimental Controls\".",
            "     3) \"Set Voltage Correction\" to on/off. Turning this on can increase the accuracy "
            "of the Arduino's voltage measurements but adds a small amount of time to each measurement.",
            "     4) \"Add New LED Color\" with the corresponding wavelength or range. If the wavelength is unknown "
            "and you wish to use the experimentally detemined value for Planck's constant to calculate the "
            "LED's wavelength, enter in \"???\" for the wavelength; ignore the quotes for entries.",
            "     5) Select the desired \"Experiment LED Color\" and \"Run Experiment\".",
            "     6) Replace the LED with a different one and repeat steps 4 and 5 until all data is aquired. "
            "Only \"Add New LED Color\" if the LED you are using has not been entered before. "
            "Ensure that you have selected the \"Experiment LED Color\" before you start the experiment, "
            "as results are automatically saved for the selected color. If no color has been selected, the selection menu "
            "is blank, then the experiment can run and results will be displayed but they will not be saved.",
            "     7) \"Select Data Folder\" to set where data will be saved or loaded from and \"Save Experimental Results\".",
            "-----------------------",
            "If you are working with saved data then:",
            "     1) Navigate to \"Experimental Controls\".",
            "     2) \"Select Data Folder\" and \"Load Experimental Results\".",
            "",
            "",
            "-----------------------",
            "Data Processing:",
            "-----------------------",
            "     1) Navigate to \"Experimental Analysis\".",
            "     2) To see the results for a single LED color, select the color from \"Select Analysis Results\". "
            "All results where the LED wavelength is known are provided first. If an LED wavelength was "
            "provided as unknown, \"???\", it will be listed after the other colors with a bar in the "
            "menu with a bar seperating it from the rest.",
            "     3) To view the cumulative results, from \"Select Analysis Results\" select "
            "\"Calculate Planck's Constant\".",
            "     4) To obtain the calculated wavelength for an LED with an unknown wavelength, select the desired "
            "LED color from \"Select Analysis Results\". Note: You must run \"Calculate Planck's Constant\" first as the "
            "value that is obtained from this experiment is used to calculate this wavelength.",
            "     5) At anytime you may \"Save Current Figure\" into the folder selected in \"Experimental Controls\" using "
            "the \"Select Data Folder\" button. "
            "Note: The size and aspect ratio of the saved figure will be the same as that shown on screen."
            ])
        self.intro_text = QPlainTextEdit(readOnly=True, plainText = self.intro_instructions)
        self.intro_text.backgroundVisible = False
        self.intro_text.wordWrapMode = True
        self.intro_text.zoomIn(2)
        self.instruction_tabs.addTab(self.intro_text, "Instruction Text")
        
        self.label = QLabel(self)
        self.image_file = os.path.join(os.getcwd(), 'Plancks Constant_schematic.png')
        self.circuit_image = QPixmap(self.image_file)
        
        self.label.setPixmap(self.circuit_image) 
  
        # Optional, resize label to image size 
        #self.label.resize(self.pixmap.width(), 
        #                  self.pixmap.height()) 
        
        self.instruction_tabs.addTab(self.label, "Circuit Diagram")
        
        
        
        
        self.layout.addWidget(self.instruction_tabs, 1, 0, 1, 4)
        
        self.setLayout(self.layout)
        
        self.search_for_devices()
    
    def search_for_devices(self) :
        """
        Runs test to find ports that Arduinos are connected to.

        Returns
        -------
        None.

        """
        self.avail_ports = get_avail_ports()
        
        self.ports_selection.clear()
        if len(self.avail_ports) > 0 and self.parent.experiment.arduino is None :
            for port in self.avail_ports :
                self.ports_selection.addItem( port )
            self.connect_btn.setEnabled(True)
        else :
            self.connect_btn.setEnabled(False)
    
    def connect_to_exp(self) :
        """
        Establishes connect to Arduino.

        Returns
        -------
        None.

        """
        self.connect_btn.setEnabled(False)
        port = self.avail_ports[ self.ports_selection.currentIndex() ]
        self.parent.setup_experiment(port)

class exp_controls(QWidget) :
    """
    Experimental controls widget.
    Allows the user to add LED color and wavelengths.
    The experiment is run here and the results are displayed with a plot.
    """
    def __init__(self, parent) :
        """
        Constructs exp_controls class.

        Parameters
        ----------
        parent : MainWindow object
            MainWindow object that this object belongs to.

        Returns
        -------
        None.

        """
        super(QWidget, self).__init__(parent)
        
        max_widget_width = 250
        
        self.parent = parent
        
        self.LED_colors = []
        
        self.layout = QGridLayout(self) # plot and progress bar
        
        self.data_plot = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout.addWidget(self.data_plot, 0, 0, 5, 1)
        
        self.exp_prog_bar = QProgressBar()
        self.layout.addWidget(self.exp_prog_bar, 5, 0, 1, 2)
        
        
        self.control_layout = QGridLayout() # controls to run experiment
        row = 0
        
        label = QLabel("")
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        label_txt = "Set Voltage Correction:"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.voltage_corr = QComboBox(self)
        self.voltage_corr.addItem( "off" )
        self.voltage_corr.addItem( "on" )
        self.voltage_corr.currentIndexChanged.connect(self.update_voltage_correction)
        self.control_layout.addWidget(self.voltage_corr, row, 0); row += 1
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        label_txt = "Add new LED color"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        label_txt = "Color Label (ex. red)"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.color_label = QLineEdit()
        self.color_label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(self.color_label, row, 0); row += 1
        
        label_txt = "Wavelength Range [nm] (ex. 460-465)"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        label_txt = "Note: values must be integers"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.color_wavelength_rng = QLineEdit()
        self.color_wavelength_rng.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(self.color_wavelength_rng, row, 0); row += 1
        
        self.add_new_led = QPushButton("Add New LED Color")
        self.add_new_led.clicked.connect(self.add_new_led_color)
        self.control_layout.addWidget(self.add_new_led, row, 0); row += 1
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        label_txt = "Experiment LED Color"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.exp_led_color = QComboBox(self)
        self.exp_led_color.addItem("")
        self.control_layout.addWidget(self.exp_led_color, row, 0); row += 1
        
        self.run_exp_btn = QPushButton("Run Experiment")
        self.run_exp_btn.setEnabled(False)
        self.run_exp_btn.setMaximumWidth( max_widget_width )
        self.run_exp_btn.clicked.connect(self.run_experiment)
        self.control_layout.addWidget(self.run_exp_btn, row, 0); row += 1
        
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        label = QLabel("")
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.btn_get_folder = QPushButton("Select Data Folder")
        self.btn_get_folder.setMaximumWidth( max_widget_width )
        self.btn_get_folder.clicked.connect(self.get_save_folder)
        self.control_layout.addWidget(self.btn_get_folder, row, 0); row += 1
        
        self.btn_load = QPushButton("Load Experimental Results")
        self.btn_load.setMaximumWidth( max_widget_width )
        self.btn_load.clicked.connect(self.load_data)
        self.control_layout.addWidget(self.btn_load, row, 0); row += 1
        
        self.btn_save = QPushButton("Save Experimental Results")
        self.btn_save.setMaximumWidth( max_widget_width )
        self.btn_save.clicked.connect(self.save_data)
        self.control_layout.addWidget(self.btn_save, row, 0); row += 1
        
        self.layout.addLayout(self.control_layout, 0, 1)
        self.setLayout(self.layout)
    
    def enable_exp(self) :
        """
        Enables the run experiment button.

        Returns
        -------
        None.

        """
        self.run_exp_btn.setEnabled(True)
    
    def run_experiment(self) :
        """
        Runs the experiment.

        Returns
        -------
        None.

        """
        #self.parent.experiment.current_data = []
        self.disable_controls()
        led_color = self.exp_led_color.currentText()
        self.running_exp = run_experiment(experiment=self.parent.experiment, canvas=self.data_plot)
        self.running_exp.notifyProgress.connect(self.exp_prog_update)
        self.running_exp.start()
        self.running_exp.finished.connect(lambda: self.get_current_data(led_color))
    
    def exp_prog_update(self, i) :
        """
        Updates the progress bar to i%.

        Parameters
        ----------
        i : float
            Percent the progress bar should display.

        Returns
        -------
        None.

        """
        self.exp_prog_bar.setValue( i )
    
    def get_current_data(self, led_color=None) :
        """
        Runs process to save current experimental results to data frame.

        Parameters
        ----------
        led_color : str, optional
            Color of the LED. The default is None.

        Returns
        -------
        None.

        """
        if led_color is None :
            led_color = self.exp_led_color.currentText()
        
        #self.current_results = self.parent.experiment.current_data
        self.update_plot()
        
        self.parent.experiment.add_exp_data(led_color)
        
        self.enable_controls()
        
    def disable_controls(self) :
        """
        Disables the experiment controls.

        Returns
        -------
        None.

        """
        self.voltage_corr.setEnabled(False)
        self.add_new_led.setEnabled(False)
        self.exp_led_color.setEnabled(False)
        self.run_exp_btn.setEnabled(False)
        self.btn_get_folder.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.parent.main_tabs.setTabEnabled(0, False)
        self.parent.main_tabs.setTabEnabled(2, False)
    
    def enable_controls(self) :
        """
        Enables the experiment controls.

        Returns
        -------
        None.

        """
        self.voltage_corr.setEnabled(True)
        self.add_new_led.setEnabled(True)
        self.exp_led_color.setEnabled(True)
        self.run_exp_btn.setEnabled(True)
        self.btn_get_folder.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.parent.main_tabs.setTabEnabled(0, True)
        self.parent.main_tabs.setTabEnabled(2, True)
    
    def update_plot(self) :
        """
        Updates the plot with current data.

        Returns
        -------
        None.

        """
        self.data_plot.axes.cla()
        
        current_data = self.parent.experiment.current_data
        xdata = np.arange(len(current_data))*5.0/4095
        
        fontsize = 15
        self.data_plot.axes.plot(xdata, current_data, 'r')
        self.data_plot.axes.set_xlabel('Supplied Voltage [V]', fontsize=fontsize)
        self.data_plot.axes.set_ylabel('Voltage Across Resistor [V]', fontsize=fontsize)
        self.data_plot.axes.set_xticklabels(self.data_plot.axes.get_xticks(), fontsize=fontsize)
        self.data_plot.axes.set_yticklabels(self.data_plot.axes.get_yticks(), fontsize=fontsize)
        self.data_plot.fig.tight_layout()
        
        self.data_plot.draw()
    
    def update_voltage_correction(self) :
        """
        Updates voltage correction based on user input.

        Returns
        -------
        None.

        """
        status = self.voltage_corr.currentText()
        if status == 'on' :
            self.parent.experiment.turn_on_Vcc_correction()
        elif status == 'off' :
            self.parent.experiment.turn_off_Vcc_correction()
    
    def add_new_led_color(self) :
        """
        Adds a new LED color and wavelength based on user input.

        Returns
        -------
        None.

        """
        label = self.color_label.text()
        wavelength_rng = self.color_wavelength_rng.text()
        
        w_rng = wavelength_rng.split('-')
        
        warning_msg = '\n'.join(["Wavelength entries must be interger values with units nanometers, nm.", 
            "Wavelength options can entered in three different way.", 
            "     1) Single wavelength to be used: \"435\"", 
            "     2) Wavelength range: \"435-450\" (An average of these values will be used for calculations)", 
            "     3) If the wavelength of the LED is to be calculated used the experimentally obtained", 
            "        value for Planck's constant, enter \"???\".", 
            "Note: exclude quotes when entering any values."])
        
        if label in self.parent.experiment.colors :
            title = "Color Label Error"
            warning_msg = "Color labels can only be used once.\n" + warning_msg
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        elif len(w_rng)==1 and w_rng[0].isdigit() or \
               len(w_rng)==1 and w_rng[0]=='???' or \
               len(w_rng)==2 and w_rng[0].isdigit() and w_rng[1].isdigit() :
                self.color_label.setText("")
                self.color_wavelength_rng.setText("")
                
                self.exp_led_color.addItem( label )
                
                self.parent.experiment.add_color(label, wavelength_rng)
                
                self.exp_led_color.setCurrentIndex( self.exp_led_color.count()-1 )
                self.parent.added_new_color = True
        else :
            title = "Value Entry Error"
            warning_msg = "An error occured with the entry values for either color or wavelength.\n" + warning_msg
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
    
    def get_save_folder(self) :
        """
        Gets the location of the save folder.

        Returns
        -------
        None.

        """
        folder = os.getcwd()
        if self.parent.experiment.save_folder is not None :
            folder = self.parent.experiment.save_folder
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", folder)
        self.parent.experiment.save_folder = folder
    
    def save_data(self) :
        """
        Initializes the save data process.

        Returns
        -------
        None.

        """
        if self.parent.experiment.save_folder is None :
            title = "Save Error"
            warning_msg = '\n'.join(["No folder has been selected to save data.",
                                     "Please select a folder."])
            warning_window = warningWindow(self)
            warning_window.build_window(title, warning_msg)
        else :
            self.parent.experiment.process_results( self.exp_prog_bar )
            self.parent.experiment.dump_dfs()
            self.parent.experiment.save_dfs()
    
    def load_data(self) :
        """
        Initializes the load data process.

        Returns
        -------
        None.

        """
        data_exists = False
        for clr in self.parent.experiment.colors :
            if not self.parent.experiment.dfs[clr].empty :
                data_exists = True
                break
        
        if data_exists :
            title = "Load Data Conformation"
            warning_msg = "\n".join(["Previous data already exists.", 
                                     "New data cannot be loaded as it", 
                                     "will over write the previous data."])
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        elif self.parent.experiment.save_folder is None :
            title = "Load Error"
            warning_msg = '\n'.join(["No folder has been selected to load data.",
                                     "Please select a folder."])
            warning_window = warningWindow(self)
            warning_window.build_window(title, warning_msg)
        else :
            self.parent.experiment.load_dfs( self.exp_prog_bar )
            self.exp_led_color.clear()
            self.exp_led_color.addItem("")
            for clr in self.parent.experiment.colors :
                self.exp_led_color.addItem( clr )
            self.parent.added_new_color = True

class results_tab(QWidget) :
    """
    Results tab widget.
    Displays the results for the knee voltage of individual LEDs and the results
        for Planck's constant.
    """
    def __init__(self, parent) :
        """
        Initializes the results_tab class.

        Parameters
        ----------
        parent : MainWindow object
            MainWindow object that this object belongs to.

        Returns
        -------
        None.

        """
        super(QWidget, self).__init__(parent)
        self.parent = parent
        max_widget_width = 150
        self.current_fig = ""
        
        self.layout = QGridLayout(self) # plot and progress bar
        
        self.data_plot = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout.addWidget(self.data_plot, 0, 0, 5, 1)
        
        self.results_prog_bar = QProgressBar()
        self.layout.addWidget(self.results_prog_bar, 5, 0, 1, 2)
        
        self.control_layout = QGridLayout()
        row = 0
        
        label = QLabel("")
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        label_txt = "Analysis Results"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        label_txt = "Select LED Color Results:"
        label = QLabel(label_txt)
        label.setMaximumWidth( max_widget_width )
        self.control_layout.addWidget(label, row, 0); row += 1
        
        self.qcb_clrs = QComboBox(self)
        self.qcb_clrs.currentIndexChanged.connect(self.selected_color)
        self.control_layout.addWidget(self.qcb_clrs, row, 0); row += 1
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        self.qbtn_calc_planck = QPushButton("Calculate Planck's Constant")
        self.qbtn_calc_planck.clicked.connect(self.plot_plancks_constant)
        self.qbtn_calc_planck.setEnabled(False)
        self.control_layout.addWidget(self.qbtn_calc_planck, row, 0); row += 1
        
        self.control_layout.addWidget(QHLine(), row, 0); row += 1
        
        self.btn_save_fig = QPushButton("Save Current Figure")
        self.btn_save_fig.clicked.connect(self.save_current_fig)
        self.control_layout.addWidget(self.btn_save_fig, row, 0); row += 1
        
        self.layout.addLayout(self.control_layout, 0, 1)
    
    def update_clr_list(self) :
        """
        Updates the list of colors for the drop down menu.

        Returns
        -------
        None.

        """
        num_clrs = len(self.parent.experiment.colors)
        
        if self.parent.added_new_color :
            self.qcb_clrs.clear()
            unknown_clrs = []
            for clr in self.parent.experiment.colors :
                if self.parent.experiment.wavelengths[clr] == '???' :
                    unknown_clrs.append(clr)
                    continue
                self.qcb_clrs.addItem(clr)
            if self.qcb_clrs.count() > 1 :
                self.qbtn_calc_planck.setEnabled(True)
            else :
                self.qbtn_calc_planck.setEnabled(False)
            if len(unknown_clrs) > 0 :
                self.qcb_clrs.insertSeparator(self.qcb_clrs.count())
                for clr in unknown_clrs :
                    self.qcb_clrs.addItem(clr)
            self.parent.added_new_color = False
        if self.current_fig != '' :
            if self.current_fig == 'Plancks Constant Results' :
                self.plot_plancks_constant()
            elif self.parent.experiment.clr_df_updated[self.current_fig] :
                self.qcb_clrs.setCurrentText(self.current_fig)
                self.parent.experiment.process_results(self.current_fig)
                self.parent.experiment.create_LED_result_plot(self.data_plot, self.current_fig)
    
    def selected_color(self) :
        """
        Gets the user selected color.

        Returns
        -------
        None.

        """
        clr = self.qcb_clrs.currentText()
        if clr == '' :
            pass
        else :
            self.current_fig = clr
            self.parent.experiment.process_results(clr)
            self.parent.experiment.create_LED_result_plot(self.data_plot, clr)
    
    def save_current_fig(self) :
        """
        Saves the currently displayed data plot to file.

        Returns
        -------
        None.

        """
        if self.parent.experiment.save_folder is None :
            title = "Save Figure Error"
            warning_msg = '\n'.join(["No folder has been selected to save data.",
                "Please return to the \"Experimental Controls\" tab to select a folder."])
            warning_window = warningWindow(self)
            warning_window.build_window(title, warning_msg)
        else :
            clr = self.qcb_clrs.currentText()
            self.parent.experiment.save_LED_figure_result(self.data_plot, clr)
    
    def plot_plancks_constant(self) :
        """
        Creates a plot of the knee voltage results and fit used to obtain
        Planck's constant.

        Returns
        -------
        None.

        """
        self.qcb_clrs.setCurrentIndex(-1)
        self.current_fig = 'Plancks Constant Results'
        self.calc_plancks_constant()
    
    def calc_plancks_constant(self) :
        """
        Runs the process to calculate Planck's constant.

        Returns
        -------
        None.

        """
        self.parent.experiment.process_results(progress_bar=self.results_prog_bar)
        self.parent.experiment.calc_Plancks_constant()
        self.parent.experiment.plot_knee_voltages(self.data_plot)
        










class QHLine(QFrame):
    """
    """
    def __init__(self):
        """
        Plots a horizontal line across the GUI.

        Returns
        -------
        None.

        """
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class QVLine(QFrame):
    """
    """
    def __init__(self):
        """
        Plots a vertical line across the GUI.dssdfsdfsdfsdf

        Returns
        -------
        None.

        """
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)











class MplCanvas(FigureCanvas) :
    """
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """

        Parameters
        ----------
        parent : object, optional
            This is for the parent object. The default is None.
        width : int, optional
            Sets the width of the canvas. The default is 5.
        height : int, optional
            Sets the height of the canvas. The default is 4.
        dpi : int, optional
            Sets the dpi of the canvas. The default is 100.

        Returns
        -------
        None.

        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)




class warningWindow(QDialog):
    """
    Default pop up window to display warnings.
    There are no buttons or selectable options, this window is to display
        simple information to the user and then have the user close it.
    """
    def __init__(self, *args, **kwargs):
        """
        

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(warningWindow, self).__init__(*args, **kwargs)
        self.title = ''
        self.msg = ''
    
    def set_title(self, title) :
        """
        Set the window's title.

        Parameters
        ----------
        title : str
            Window title.

        Returns
        -------
        None.

        """
        self.title = title
    
    def set_msg(self, msg) :
        """
        Set the window's main message.

        Parameters
        ----------
        msg : str
            Window message.

        Returns
        -------
        None.

        """
        self.msg = msg
    
    def set_text_msgs(self, title, msg) :
        """
        Set window text, title and main message.

        Parameters
        ----------
        title : str
            Window title.
        msg : str
            Window message.

        Returns
        -------
        None.

        """
        self.title = title
        self.msg = msg
    
    def build_window(self, title=None, msg=None) :
        """
        This will create the window and display it.

        Parameters
        ----------
        title : str
            Window title.
        msg : str
            Window message.

        Returns
        -------
        None.

        """
        if title is not None :
            self.title = title
        if msg is not None :
            self.msg = msg
        
        self.setWindowTitle(self.title)
        
        QBtn = QDialogButtonBox.Ok # | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        
        label = QLabel(self.msg)
        self.layout.addWidget(label)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        self.exec_()



if __name__ == '__main__' :
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
    window.experiment.disconnect_from_arduino()
    sys.exit()



