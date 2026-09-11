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
  <FrameText0=Butterfly radial stub band-stop (2-port notch)>
  <FrameText1=Drawn By: qucs-llm-optimizer>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <GND * 1 80 280 0 0 0 0>
  <Pac P1 1 80 230 18 -26 0 1 "1" 1 "50 Ohm" 1 "0 dBm" 0 "5000000000.0Hz" 0 "26.85" 0 "true" 0>
  <MLIN MLin 1 180 200 -26 15 0 0 "Subst1" 1 "0.359mm" 1 "8.6415mm" 1 "Hammerstad" 0 "Kirschning" 0 "26.85" 0 "DC" 0>
  <MCROSS J1 1 240 200 -40 -40 0 0 "Subst1" 1 "0.359mm" 1 "0.359mm" 1 "0.359mm" 1 "0.359mm" 1 "Hammerstad" 0 "Kirschning" 0 "showNumbers" 0>
  <MRSTUB MSW1 1 240 160 -30 20 0 0 "Subst1" 0 "1.256mm" 1 "5.7094mm" 1 "0.359mm" 1 "78.4404" 1 "OldQucsNoCorrection" 0 "OldQucsModel" 0>
  <MRSTUB MSW2 1 240 240 -30 -50 1 0 "Subst1" 0 "1.256mm" 1 "5.7094mm" 1 "0.359mm" 1 "78.4404" 1 "OldQucsNoCorrection" 0 "OldQucsModel" 0>
  <MLIN MLout 1 300 200 -26 15 0 0 "Subst1" 1 "0.359mm" 1 "8.6415mm" 1 "Hammerstad" 0 "Kirschning" 0 "26.85" 0 "DC" 0>
  <Pac P2 1 400 230 18 -26 0 1 "2" 1 "50 Ohm" 1 "0 dBm" 0 "5000000000.0Hz" 0 "26.85" 0 "true" 0>
  <GND * 1 400 280 0 0 0 0>
  <.SP SP1 1 50 40 0 45 0 0 "lin" 1 "1000000000.0Hz" 1 "10000000000.0Hz" 1 "181" 1 "no" 0 "1" 0 "2" 0 "no" 0 "no" 0>
  <SUBST Subst1 1 120 60 -30 24 0 0 "3.38" 1 "0.508mm" 1 "0.035mm" 1 "0.0027" 1 "1.72e-08" 1 "1.5e-07" 1>
  <Eqn Eqn1 1 50 360 -28 15 0 0 "Zin_norm=abs(rtoz(S[1,1]))" 1 "Zin_ohm=50*Zin_norm" 1 "S11_dB=dB(S[1,1])" 1 "S21_dB=dB(S[2,1])" 1 "S11_phase=phase(S[1,1])" 1 "S21_phase=phase(S[2,1])" 1 "yes" 0>
</Components>
<Wires>
  <80 200 150 200 "" 0 0 0 "">
  <330 200 400 200 "" 0 0 0 "">
  <80 260 80 280 "" 0 0 0 "">
  <400 260 400 280 "" 0 0 0 "">
</Wires>
<Diagrams>
  <Rect 520 40 340 200 3 #c0c0c0 1 00 0 1000000000.0 1e+09 10000000000.0 1 0 1 20 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "|Zin|/Z0 at P1" "">
	<"Zin_norm" #ff0000 0 3 0 0 0>
  </Rect>
  <Rect 520 280 340 200 3 #c0c0c0 1 00 0 1000000000.0 1e+09 10000000000.0 1 -60 10 5 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S11 / S21, dB" "">
	<"S11_dB" #ff0000 0 3 0 0 0>
	<"S21_dB" #0000ff 0 3 0 0 0>
  </Rect>
  <Rect 520 520 340 200 3 #c0c0c0 1 00 0 1000000000.0 1e+09 10000000000.0 1 -180 45 180 1 -1 1 1 315 0 225 1 0 0 "Frequency, Hz" "S11 / S21 phase, deg" "">
	<"S11_phase" #ff0000 0 3 0 0 0>
	<"S21_phase" #0000ff 0 3 0 0 0>
  </Rect>
  <Smith 900 160 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[1,1]" #ff0000 0 3 0 0 0>
  </Smith>
  <Polar 1200 160 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[1,1]" #008000 0 3 0 0 0>
  </Polar>
  <Polar 900 480 260 260 3 #c0c0c0 1 00 1 0 1 1 1 0 4 1 1 0 1 1 315 0 225 "" "" "">
	<"S[2,1]" #0000ff 0 3 0 0 0>
  </Polar>
</Diagrams>
<Paintings>
  <Text 300 10 12 #000080 0 "Band-stop / notch: shunt butterfly shorts the through path in-band\nStopband target 4-6 GHz: minimize max |S21| (deep notch)\nS11 charts (Zin/Smith/Polar) + S21 dB/phase/Polar\nJunction is MCROSS (required by Qucs-RFlayout)">
  <Text 1180 460 12 #0000ff 0 "Polar S21">
</Paintings>
