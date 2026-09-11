<Qucs Schematic 25.2.0>
<Properties>
  <View=0,0,1500,1000,0.8,0,0>
  <Grid=10,10,1>
  <DataSet=bpf_butterworth3_cf100M_bw5M.dat>
  <DataDisplay=bpf_butterworth3_cf100M_bw5M.dpl>
  <OpenDisplay=1>
  <Script=bpf_butterworth3_cf100M_bw5M.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=3rd-order Butterworth bandpass, f0 = 100 MHz, BW = 5 MHz, Z0 = 50 Ohm>
  <FrameText1=Drawn By:>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <Pac P1 1 100 260 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "100 MHz" 0 "26.85" 0>
  <GND * 1 100 290 0 0 0 0>
  <L L1 1 200 180 -26 -48 0 0 "1591.5494nH" 1 "" 0>
  <C C1 1 280 180 -26 -48 0 0 "1.5915494pF" 1 "" 0 "neutral" 0>
  <C C2 1 380 240 17 -26 0 1 "1273.2395pF" 1 "" 0 "neutral" 0>
  <GND * 1 380 270 0 0 0 0>
  <L L2 1 460 240 10 -26 0 1 "1.9894368nH" 1 "" 0>
  <GND * 1 460 270 0 0 0 0>
  <L L3 1 560 180 -26 -48 0 0 "1591.5494nH" 1 "" 0>
  <C C3 1 640 180 -26 -48 0 0 "1.5915494pF" 1 "" 0 "neutral" 0>
  <Pac P2 1 740 260 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "100 MHz" 0 "26.85" 0>
  <GND * 1 740 290 0 0 0 0>
  <.SP SP1 1 100 400 0 45 0 0 "lin" 1 "50 MHz" 1 "150 MHz" 1 "1001" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <Eqn Eqn1 1 300 400 -28 15 0 0 "S21_dB=dB(S[2,1])" 1 "S11_dB=dB(S[1,1])" 1 "GD_ns=1e9*groupdelay(S,2,1)" 1 "yes" 0>
</Components>
<Wires>
  <100 180 100 230 "" 0 0 0 "">
  <100 180 170 180 "" 0 0 0 "">
  <230 180 250 180 "" 0 0 0 "">
  <310 180 380 180 "" 0 0 0 "">
  <380 180 380 210 "" 0 0 0 "">
  <380 180 460 180 "" 0 0 0 "">
  <460 180 460 210 "" 0 0 0 "">
  <460 180 530 180 "" 0 0 0 "">
  <590 180 610 180 "" 0 0 0 "">
  <670 180 740 180 "" 0 0 0 "">
  <740 180 740 230 "" 0 0 0 "">
</Wires>
<Diagrams>
  <Rect 880 300 420 260 3 #c0c0c0 1 00 0 50000000.0 1e+07 150000000.0 1 -80 10 0 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S21 / S11, dB" "">
	<"S21_dB" #0000ff 0 3 0 0 0>
	<"S11_dB" #ff0000 0 3 0 0 0>
  </Rect>
  <Rect 880 620 420 260 3 #c0c0c0 1 00 0 95000000.0 1e+06 105000000.0 1 -20 2 0 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S21 / S11, dB (zoom)" "">
	<"S21_dB" #0000ff 0 3 0 0 0>
	<"S11_dB" #ff0000 0 3 0 0 0>
  </Rect>
  <Smith 880 960 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[1,1]" #ff0000 0 3 0 0 0>
  </Smith>
  <Rect 1240 1000 300 220 3 #c0c0c0 1 00 0 95000000.0 2e+06 105000000.0 1 0 50 200 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "Group delay, ns" "">
	<"GD_ns" #008000 0 3 0 0 0>
  </Rect>
</Diagrams>
<Paintings>
  <Text 100 30 12 #000080 0 "3rd-order Butterworth bandpass filter\nf0 = 100 MHz, 3 dB BW = 5 MHz (fractional BW D = 0.05), Z0 = 50 Ohm\n\nLowpass prototype (n = 3, Butterworth): g1 = 1.0 (series), g2 = 2.0 (shunt), g3 = 1.0 (series)\n\nSeries resonator (from series g):   Ls = g*Z0/(D*w0)      Cs = D/(g*Z0*w0)\nShunt  resonator (from shunt  g):   Cp = g/(D*Z0*w0)      Lp = D*Z0/(g*w0)\n\nL1 = L3 = 1591.5494 nH   C1 = C3 = 1.5915494 pF   (series, resonate at 100 MHz)\nC2 = 1273.2395 pF        L2 = 1.9894368 nH        (shunt,  resonate at 100 MHz)\n\nEvery branch resonates at f0: series branches become a short, the shunt branch\nbecomes an open, so at 100 MHz the filter is transparent (S21 = 0 dB, S11 = -inf).\nOff resonance the series branches block and the shunt branch loads the line,\nwhich produces the 3rd-order (-18 dB/octave) skirts.\n\nComponents are ideal (infinite Q). Real inductors with Q ~ 60 cost roughly\n1-2 dB of insertion loss at this 5 MHz bandwidth.">
</Paintings>
