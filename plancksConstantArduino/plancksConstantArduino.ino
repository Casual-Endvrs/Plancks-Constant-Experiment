//  This program will listen to the serial port for instructions
//  When told to 'run' the arduino will step wise progress from
//    0V to Vcc using a DAC. The DAC will power an LED. After each
//    time the voltage is advanced a voltage measurement of a resistor
//    will be taken. The resistor will be in series with the LED and 
//    located between the LED and GND.
//    V_DAC --> LED --> (measure point) --> resistor --> GND
//  Measured voltage correction: (toggle-able feature)
//    This method is added in the case where voltage correction is implimented.
//    If the bool variable 'correct_Vcc' is true, the correction method will be used.
//    The measured voltage may vary due to Vcc varying over a small range.
//    This error will be corrected using a method from:
//      https://www.youtube.com/watch?v=xI_qU2auVx8
//      https://github.com/SensorsIot/ADC_Test/blob/master/ADC_Test.ino
//    Alternative source - see section Analog Readings:
//      https://www.codeproject.com/tips/987180/arduino-tips-tricks
//    Eqs.
//      Vcc_corrected = 1.1 * 1023 * 1000 / Vcc_measured
//      V_corrected = Vcc_corrected / Vcc * V_measured
//      where:
//        Vcc (float, const) - voltage provided from Arduino to DAC, 3.3V or 5V
//        Vcc_measured (int, 0-1024) - voltage supplied to DAC as measured by A0
//        Vcc_corrected (float, 0-Vcc) - corrected voltage that is supplied to DAC
//        V_measured (int, 0-1024) - voltage at (measure point) as measured by A1
//        V_corrected (float, 0-Vcc) - corrected voltage at (measure point)
//  Pins:
//    A0 - Analogue, measures Vcc using reference ####################
//    A1 - Analogue, measures voltage at (measure point)
//    A4 - I2C, SDA - serial data
//    A5 - I2C, SCL - serial clock
//  Equipment:
//    LED - various as many different wavelength LEDs are needed
//      for measurements. To be chosen by user.
//    Resistor - 
//    DAC - MCP4725 - https://www.adafruit.com/product/935
//      Vcc - This can be 3.3V or 5V depending on the voltage
//        requirements of the LED in use.

#include <Wire.h> // used for I2C communication
#include <Adafruit_MCP4725.h> // MCP4725 DAC

Adafruit_MCP4725 dac;

// define pins
const int measure_led_v = A0;

// create variables
float Vcc = 5.;   // set to 3.3 or 5. depending on voltage supplied to DAC
float V_control_measured;
float Vcc_corrected;
float V_measured;

String serial_command;

bool correct_Vcc = false;

void setup(){
  // open serial port
  Serial.begin(9600);
  
  // initiate DAC
  dac.begin(0x61);
  
  // set DAC voltage to 0
  dac.setVoltage(0, false);
  
  pinMode(13, OUTPUT);
}

void loop(){
  // wait for a Serial command to be available
  while (!Serial.available()) {}
  while (Serial.available()) {
    serial_command = Serial.readStringUntil('/');
  }
  
  // if command == 'CVcc,t' or 'CVcc,f' - set correct_Vcc = t/f
  // This allows the python program to turn on/off voltage measurement correction
  if (serial_command=="CVcc,t") { correct_Vcc = true; }
  else if (serial_command=="CVcc,f") { correct_Vcc = false; }
  else if (serial_command=="run") {
    digitalWrite(13, HIGH);
    // start loop to iter through 4096 DAC voltage range
    for (int i=0; i<4096; i++){
      // set DAC value
      dac.setVoltage(i, false);
      // if correct_Vcc
      if (correct_Vcc) {
        // measure & calculate Vcc_corrected
        V_control_measured = readVcc() / 1000.0;
        // measure (measure point), calculate percent of Vcc
        V_measured = analogRead( measure_led_v ) / 1023.;
        // calculate corrected voltage, return result over serial
        Serial.println( V_control_measured * V_measured );
        // Serial.println( String(V_control_measured * V_measured) + ',' );
      }
      else {
        // delay 2 ms to allow all voltages to settle
        delay(2);
        // measure (measure point)
        V_measured = analogRead( measure_led_v ) / 1023.;
        // Serial return V_measured
        Serial.println( Vcc * V_measured );
        // Serial.println( String(Vcc * V_measured) + ',' );
      }
    }
    // return 'end' to indicate experiment is complete
    Serial.println( "end" );
    digitalWrite(13, LOW);
    // set DAC voltage to 0
    dac.setVoltage(0, false);
  }
}

long readVcc() {
  long result;
  // Read 1.1V reference against AVcc
#if defined(__AVR_ATmega32U4__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega2560__)
  ADMUX = _BV(REFS0) | _BV(MUX4) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
#elif defined (__AVR_ATtiny24__) || defined(__AVR_ATtiny44__) || defined(__AVR_ATtiny84__)
  ADMUX = _BV(MUX5) | _BV(MUX0);
#elif defined (__AVR_ATtiny25__) || defined(__AVR_ATtiny45__) || defined(__AVR_ATtiny85__)
  ADMUX = _BV(MUX3) | _BV(MUX2);
#else
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
#endif
  delay(2); // Wait for Vref to settle
  ADCSRA |= _BV(ADSC); // Convert
  while (bit_is_set(ADCSRA, ADSC));
  result = ADCL;
  result |= ADCH << 8;
  result = 1126400L / result; // Calculate Vcc (in mV); 1126400 = 1.1*1024*1000
  return result;
}
