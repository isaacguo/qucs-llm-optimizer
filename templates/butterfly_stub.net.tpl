# Butterfly (bowtie) radial stub - 1-port shunt bias-decoupling test circuit
# Two MRSTUB "wings" tied to the same node model the symmetric butterfly/bowtie
# fan shape as two parallel 1-port radial-stub admittances (circuit-level
# approximation; verify with full-wave EM before fabrication).
Pac:P1 nodeA gnd Num="1" Z="50 Ohm" P="0 dBm" f="{f0_hz}Hz"
MLIN:MLc nodeA nodeB Subst="Subst1" W="{wf_mm}mm" L="{lc_mm}mm" Model="Hammerstad" DispModel="Kirschning"
MRSTUB:MSW1 nodeB Subst="Subst1" ri="{ri_mm}mm" ro="{ro_mm}mm" Wf="{wf_mm}mm" alpha="{alpha_deg}" EffDimens="OldQucsNoCorrection" Model="OldQucsModel"
MRSTUB:MSW2 nodeB Subst="Subst1" ri="{ri_mm}mm" ro="{ro_mm}mm" Wf="{wf_mm}mm" alpha="{alpha_deg}" EffDimens="OldQucsNoCorrection" Model="OldQucsModel"
.SP:SP1 Type="lin" Start="{sweep_start_hz}Hz" Stop="{sweep_stop_hz}Hz" Points="{sweep_points}" Noise="no" NoiseIP="1" NoiseOP="2" saveCVs="no" saveAll="no"
SUBST:Subst1 er="{er}" h="{h_mm}mm" t="{t_mm}mm" tand="{tand}" rho="{rho}" D="{d_m}"
