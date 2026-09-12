<Qucs Schematic 25.2.0>
<Properties>
  <View=0,0,1700,1100,0.65,0,0>
  <Grid=10,10,1>
  <DataSet=circuit.dat>
  <DataDisplay=circuit.dpl>
  <OpenDisplay=0>
  <Script=circuit.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=5th-order Butterworth BPF (series-first ladder)>
  <FrameText1=Drawn By: qucs-llm-optimizer>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <GND * 1 80 260 0 0 0 0>
  <Pac P1 1 80 230 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "221396905.77425343Hz" 0 "26.85" 0 "true" 0>
  <L L1 1 150 200 -26 10 0 0 "346.6919 nH" 1 "" 0>
  <C C1 1 240 200 -26 17 0 0 "2.84 pF" 1 "" 0 "neutral" 0>
  <L L2 1 300 260 17 -26 0 1 "2.7119 nH" 1 "" 0>
  <C C2 1 340 260 17 -26 0 1 "363.0728 pF" 1 "" 0 "neutral" 0>
  <GND * 1 300 290 0 0 0 0>
  <GND * 1 340 290 0 0 0 0>
  <L L3 1 390 200 -26 10 0 0 "1121.9803 nH" 1 "" 0>
  <C C3 1 480 200 -26 17 0 0 "0.8776 pF" 1 "" 0 "neutral" 0>
  <L L4 1 540 260 17 -26 0 1 "2.7119 nH" 1 "" 0>
  <C C4 1 580 260 17 -26 0 1 "363.0728 pF" 1 "" 0 "neutral" 0>
  <GND * 1 540 290 0 0 0 0>
  <GND * 1 580 290 0 0 0 0>
  <L L5 1 630 200 -26 10 0 0 "346.6919 nH" 1 "" 0>
  <C C5 1 720 200 -26 17 0 0 "2.84 pF" 1 "" 0 "neutral" 0>
  <Pac P2 1 810 230 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "221396905.77425343Hz" 0 "26.85" 0 "true" 0>
  <GND * 1 810 260 0 0 0 0>
  <.SP SP1 1 50 40 0 45 0 0 "lin" 1 "0.0Hz" 1 "300000000.0Hz" 1 "301" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <Eqn Eqn1 1 50 360 -28 15 0 0 "S11_dB=dB(S[1,1])" 1 "S21_dB=dB(S[2,1])" 1 "S11_phase=phase(S[1,1])" 1 "S21_phase=phase(S[2,1])" 1 "yes" 0>
</Components>
<Wires>
  <80 200 120 200 "" 0 0 0 "">
  <180 200 210 200 "" 0 0 0 "">
  <270 200 360 200 "" 0 0 0 "">
  <300 200 300 230 "" 0 0 0 "">
  <340 200 340 230 "" 0 0 0 "">
  <420 200 450 200 "" 0 0 0 "">
  <510 200 600 200 "" 0 0 0 "">
  <540 200 540 230 "" 0 0 0 "">
  <580 200 580 230 "" 0 0 0 "">
  <660 200 690 200 "" 0 0 0 "">
  <750 200 810 200 "" 0 0 0 "">
</Wires>
<Diagrams>
  <Rect 900 40 340 200 3 #c0c0c0 1 00 0 0.0 1e+08 300000000.0 1 -60 10 5 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S11 / S21, dB" "">
	<"S11_dB" #ff0000 0 3 0 0 0>
	<"S21_dB" #0000ff 0 3 0 0 0>
  </Rect>
  <Rect 900 280 340 200 3 #c0c0c0 1 00 0 0.0 1e+08 300000000.0 1 -180 45 180 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S11 / S21 phase, deg" "">
	<"S11_phase" #ff0000 0 3 0 0 0>
	<"S21_phase" #0000ff 0 3 0 0 0>
  </Rect>
  <Smith 1280 160 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[1,1]" #ff0000 0 3 0 0 0>
  </Smith>
  <Polar 1580 160 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[1,1]" #008000 0 3 0 0 0>
  </Polar>
  <Polar 1280 480 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[2,1]" #0000ff 0 3 0 0 0>
  </Polar>
</Diagrams>
<Paintings>
  <Text 50 10 12 #000080 0 "5th-order Butterworth BPF: series-first ladder\nSeries arms L-C (L1-C1, L3-C3, L5-C5); shunt arms L||C to GND (L2||C2, L4||C4)\nS11/S21 dB + phase, Smith, Polar">
  <Text 1560 460 12 #0000ff 0 "Polar S21">
</Paintings>
