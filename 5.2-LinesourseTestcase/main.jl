using Base: Float64
include("utils.jl")
include("settings.jl")
include("SolverDLRA.jl")

using PyPlot
using DelimitedFiles

nₚₙ = 39; # nₚₙ = 39;
ϑₚ = 0.005;      # parallel tolerance
ϑᵤ = 0.05;    # (unconventional) BUG tolerance
ϑₚ₁ = 3e-3;      # parallel tolerance
ϑᵤ₁ = 2.5e-2;    # (unconventional) BUG tolerance
ϑₚ₂ = 1.5e-3;      # parallel tolerance
ϑᵤ₂ = 1e-2;    # (unconventional) BUG tolerance

s = Settings(251, 251, nₚₙ, 15); # create settings class with 251 x 251 spatial cells and a maximal rank of 100
s.ϑIndex = 1;
s.cη = 5.0;
################################################################
######################### execute code #########################
################################################################

################### run full method ###################
#solver = SolverDLRA(s);
#@time rhoFull = Solve(solver);
#Φ = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low accuracy #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time ψ,rankInTime,η,ηBound = SolveBUGAdaptiveRejection(solver);
Φᵣ = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

################### run parallel ###################
s.ϑ = ϑₚ;
solver = SolverDLRA(s);
@time ψ,rankInTimep,ηp,ηpBound = SolveParallelRejection(solver);
Φᵣp = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

##################### higher accuracy #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ₁;
solver = SolverDLRA(s);
@time ψ,rankInTime₁,η₁,η₁Bound = SolveBUGAdaptiveRejection(solver);
Φᵣ₁ = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

################### run parallel ###################
s.ϑ = ϑₚ₁;
solver = SolverDLRA(s);
@time ψ,rankInTimep₁,ηp₁,ηp₁Bound = SolveParallelRejection(solver);
Φᵣp₁ = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

##################### highest accuracy #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ₂;
solver = SolverDLRA(s);
@time ψ,rankInTime₂,η₂,η₂Bound = SolveBUGAdaptiveRejection(solver);
Φᵣ₂ = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

################### run parallel ###################
s.ϑ = ϑₚ₂;
solver = SolverDLRA(s);
@time ψ,rankInTimep₂,ηp₂,ηp₂Bound = SolveParallelRejection(solver);
Φᵣp₂ = Vec2Mat(s.NCellsX,s.NCellsY,ψ);

############################################################
######################### plotting #########################
############################################################

################### read in reference solution ###################
lsRef = readdlm("exactLineSource.txt", ',', Float64);
xRef = lsRef[:,1];
phiRef = lsRef[:,2];
lsRefFull = readdlm("refPhiFull.txt", ',', Float64);

nxRef = size(lsRefFull,1); nyRef = size(lsRefFull,2);
xRefFull = collect(range(s.a,s.b,nxRef));
yRefFull = collect(range(s.c,s.d,nxRef));

################### plot reference cut ###################
fig = figure("u cut ref",figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef,phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ylabel(L"\Phi", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=20) 
tight_layout()
show()
savefig("results/scalar_flux_reference_cut.pdf")

################### plot reference full ###################
fig = figure("scalar_flux_reference",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(xRefFull,yRefFull, lsRefFull,vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, reference", fontsize=25)
tight_layout()
savefig("results/scalar_flux_reference.png")

################### plot scalar fluxes full ###################

############################################################
######################### plotting #########################
############################################################

X = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])));
Y = (s.yMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))';
Φ = Φᵣ
## full
maxV = maximum(Φ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φ[2:end-1,2:end-1]))
fig = figure("full, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φ[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, P$_{21}$", fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_PN_$(s.problem)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(Φᵣ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣ[2:end-1,2:end-1]))
fig = figure("BUG, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣ[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(Φᵣp[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣp[2:end-1,2:end-1]))
fig = figure("parallel, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣp[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(Φᵣ₁[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣ₁[2:end-1,2:end-1]))
idxNeg = findall((Φᵣ₁.<=0.0))
Φᵣ₁[idxNeg] .= NaN;
fig = figure("BUG, log, ϑ medium",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣ₁[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₁)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(Φᵣp₁[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣp₁[2:end-1,2:end-1]))
idxNeg = findall((Φᵣp₁.<=0.0))
Φᵣp₁[idxNeg] .= NaN;
fig = figure("parallel, log, ϑ medium",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣp₁[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₁)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## highest tolerance

## DLRA BUG adaptive
maxV = maximum(Φᵣ₂[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣ₂[2:end-1,2:end-1]))
idxNeg = findall((Φᵣ₂.<=0.0))
Φᵣ₂[idxNeg] .= NaN;
fig = figure("BUG, log, ϑ fine",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣ₂[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₂)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(Φᵣp₂[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*Φᵣp₂[2:end-1,2:end-1]))
idxNeg = findall((Φᵣp₂.<=0.0))
Φᵣp₂[idxNeg] .= NaN;
fig = figure("parallel, log, ϑ fine",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*Φᵣp₂[2:(end-1),(end-1):-1:2]',vmin=0.0,vmax=maximum(lsRefFull))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₂)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

################### plot scalar fluxes cut ###################

fig = figure("Φ cut, BUG",figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef,phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣ[Int(floor(s.NCellsX/2+1)),:], "b--", linewidth=2, label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ)), alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣ₁[Int(floor(s.NCellsX/2+1)),:], "r:", linewidth=2, label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₁)), alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣ₂[Int(floor(s.NCellsX/2+1)),:], "g-.", linewidth=2, label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₂)), alpha=0.8)
ax.legend(loc="upper left", fontsize=20)
ylabel(L"\Phi", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a,s.b])
ax.set_ylim([-0.01,0.5])
ax.tick_params("both",labelsize=20) 
tight_layout()
show()
savefig("results/phi_cut_BUGnx$(s.NCellsX).pdf")

fig = figure("Φ cut, parallel",figsize=(10, 10), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
ax = gca()
ax.plot(xRef,phiRef, "k-", linewidth=2, label="exact", alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣp[Int(floor(s.NCellsX/2+1)),:], "b--", linewidth=2, label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ)), alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣp₁[Int(floor(s.NCellsX/2+1)),:], "r:", linewidth=2, label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₁)), alpha=0.8)
ax.plot(s.yMid,4.0*pi*sqrt(2)*Φᵣp₂[Int(floor(s.NCellsX/2+1)),:], "g-.", linewidth=2, label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₂)), alpha=0.8)
ax.legend(loc="upper left", fontsize=20)
ylabel(L"\Phi", fontsize=20)
xlabel(L"$x$", fontsize=20)
ax.set_xlim([s.a,s.b])
ax.set_ylim([-0.01,0.5])
ax.tick_params("both",labelsize=20) 
tight_layout()
show()
savefig("results/phi_cut_PARALLELnx$(s.NCellsX).pdf")


# plot rank in time
fig = figure("ranks",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(rankInTime[1,:],rankInTime[2,:], "b--", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimep[1,:],rankInTimep[2,:], "r--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(rankInTime₁[1,:],rankInTime₁[2,:], "b:", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₁)), linewidth=2, alpha=1.0)
ax.plot(rankInTimep₁[1,:],rankInTimep₁[2,:], "r:", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₁)), linewidth=2, alpha=1.0)
ax.plot(rankInTime₂[1,:],rankInTime₂[2,:], "b-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₂)), linewidth=2, alpha=1.0)
ax.plot(rankInTimep₂[1,:],rankInTimep₂[2,:], "r-.", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₂)), linewidth=2, alpha=1.0)
ax.set_xlim([rankInTime[1,1],rankInTime[1,end]])
ax.set_ylim([0,max(maximum(rankInTime₁[2,2:end]),maximum(rankInTimep₂[2,2:end]))+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) ;
ax.legend(loc="upper left", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

# plot η in time
fig = figure("eta coarse",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(η[:,1],η[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(ηp[:,1],ηp[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(ηBound[:,1],ηBound[:,2], "r:", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηpBound[:,1],ηpBound[:,2], "b-", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([η[1,1],η[end,1]])
ax.set_ylim([0,max(maximum(η[1:end,2]),maximum(ηp[1:end,2]),maximum(ηBound[1:end,2]),maximum(ηpBound[1:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.Nx)_N$(s.nₚₙ).png")

# plot η in time
fig = figure("eta finer",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(η₁[:,1],η₁[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₁)), linewidth=2, alpha=1.0)
ax.plot(ηp₁[:,1],ηp₁[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₁)), linewidth=2, alpha=1.0)
ax.plot(η₁Bound[:,1],η₁Bound[:,2], "r:", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηp₁Bound[:,1],ηp₁Bound[:,2], "b-", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([η₁[1,1],η₁[end,1]])
ax.set_ylim([0,max(maximum(η₁[1:end,2]),maximum(ηp₁[1:end,2]),maximum(η₁Bound[1:end,2]),maximum(ηp₁Bound[1:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_thetaBUG$(ϑᵤ₁)_thetaParallel$(ϑₚ₁)_nx$(s.Nx)_N$(s.nₚₙ).png")

# plot η in time
fig = figure("eta finest",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(η₂[:,1],η₂[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ₂)), linewidth=2, alpha=1.0)
ax.plot(ηp₂[:,1],ηp₂[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ₂)), linewidth=2, alpha=1.0)
ax.plot(η₂Bound[:,1],η₂Bound[:,2], "r:", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηp₂Bound[:,1],ηp₂Bound[:,2], "b-", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([η₂[1,1],η₂[end,1]])
ax.set_ylim([0,max(maximum(η₂[1:end,2]),maximum(ηp₂[1:end,2]),maximum(η₂Bound[1:end,2]),maximum(ηp₂Bound[1:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_thetaBUG$(ϑᵤ₂)_thetaParallel$(ϑₚ₂)_nx$(s.Nx)_N$(s.nₚₙ).png")

writedlm("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTime)
writedlm("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣ)

writedlm("results/ranks_$(s.problem)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTimep)
writedlm("results/scalar_flux_DLRA_parallel_$(s.problem)_theta$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣp)

writedlm("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTime₁)
writedlm("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣ₁)

writedlm("results/ranks_$(s.problem)_thetaParallel$(ϑₚ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTimep₁)
writedlm("results/scalar_flux_DLRA_parallel_$(s.problem)_theta$(ϑₚ₁)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣp₁)

writedlm("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTime₂)
writedlm("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣ₂)

writedlm("results/ranks_$(s.problem)_thetaParallel$(ϑₚ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", rankInTimep₂)
writedlm("results/scalar_flux_DLRA_parallel_$(s.problem)_theta$(ϑₚ₂)_nx$(s.NCellsX)_N$(s.nₚₙ).txt", Φᵣp₂)

println("main finished")
