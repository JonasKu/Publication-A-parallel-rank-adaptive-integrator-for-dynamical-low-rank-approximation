using Base: Float64
include("../5.2-LinesourseTestcase/utils.jl")
include("../5.2-LinesourseTestcase/settings.jl")
include("../5.2-LinesourseTestcase/SolverDLRA.jl")

using PyPlot
using DelimitedFiles

close("all")

nₚₙ = 21;
ϑₚ = 0.01;      # parallel tolerance ϑₚ = 2e-2; ϑᵤ = 5e-2; ϑₚₕ = 1e-2; ϑᵤₕ = 3e-2;
ϑᵤ = 0.016;    # (unconventional) BUG tolerance # BUG 0.05, parrallel 0.02 looks okay
ϑₚₕ = 0.005;      # parallel tolerance
ϑᵤₕ = 0.01;    # (unconventional) BUG tolerance

s = Settings(351,351, nₚₙ, 50,"Lattice"); # create settings class with 351 x 351 spatial cells and a rank of 50
s.cη = 5.0;
s.ϑIndex = 1;
################################################################
######################### execute code #########################
################################################################

##################### classical checkerboard #####################

################### run full method ###################
solver = SolverDLRA(s);
@time rhoFull = Solve(solver);
rhoFull = Vec2Mat(s.NCellsX,s.NCellsY,rhoFull)

##################### low tolerance #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤ;
solver = SolverDLRA(s);
@time rhoDLRA,rankInTime,η₁,η₁Bound = SolveBUGAdaptiveRejection(solver);
rhoDLRA = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRA);

################### run parallel ###################
s.ϑ = ϑₚ;
solver = SolverDLRA(s);
@time rhoDLRAp,rankInTimep, ηp₁,ηp₁Bound = SolveParallelRejection(solver);
rhoDLRAp = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAp);

##################### high tolerance #####################

################### run BUG adaptive ###################
s.ϑ = ϑᵤₕ;
solver = SolverDLRA(s);
@time rhoDLRAₕ,rankInTimeₕ,η₂,η₂Bound = SolveBUGAdaptiveRejection(solver);
rhoDLRAₕ = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRAₕ);

################### run parallel ###################
s.ϑ = ϑₚₕ;
solver = SolverDLRA(s);
@time rhoDLRApₕ,rankInTimepₕ, ηp₂,ηp₂Bound = SolveParallelRejection(solver);
rhoDLRApₕ = Vec2Mat(s.NCellsX,s.NCellsY,rhoDLRApₕ);

############################################################
######################### plotting #########################
############################################################

X = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])));
Y = (s.yMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))';

## full
maxV = maximum(rhoFull[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoFull[2:end-1,2:end-1]))
idxNeg = findall((rhoFull.<=0.0))
rhoFull[idxNeg] .= NaN;
fig = figure("full, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoFull[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, P$_{21}$", fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_PN_$(s.problem)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(rhoDLRA[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRA[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRA.<=0.0))
rhoDLRA[idxNeg] .= NaN;
fig = figure("BUG, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRA[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(rhoDLRAp[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRAp[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRAp.<=0.0))
rhoDLRAp[idxNeg] .= NaN;
fig = figure("parallel, log, ϑ coarse",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAp[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA BUG adaptive
maxV = maximum(rhoDLRAₕ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRAₕ[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRAₕ.<=0.0))
rhoDLRAₕ[idxNeg] .= NaN;
fig = figure("BUG, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRAₕ[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤₕ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_adBUG_$(s.problem)_theta$(ϑᵤₕ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

## DLRA parallel
maxV = maximum(rhoDLRApₕ[2:end-1,2:end-1])
minV = max(1e-7,minimum(4.0*pi*sqrt(2)*rhoDLRApₕ[2:end-1,2:end-1]))
idxNeg = findall((rhoDLRApₕ.<=0.0))
rhoDLRApₕ[idxNeg] .= NaN;
fig = figure("parallel, log",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoDLRApₕ[2:(end-1),(end-1):-1:2]',norm=matplotlib.colors.LogNorm(vmin=minV, vmax=maxV))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\Phi$, parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚₕ)), fontsize=25)
tight_layout()
show()
savefig("results/scalar_flux_DLRA_$(s.problem)_theta$(ϑₚₕ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

# plot rank in time
fig = figure("ranks",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(rankInTime[1,:],rankInTime[2,:], "k--", label=L"BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimep[1,:],rankInTimep[2,:], "r--", label=L"parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimeₕ[1,:],rankInTimeₕ[2,:], "k:", label=L"BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤₕ)), linewidth=2, alpha=1.0)
ax.plot(rankInTimepₕ[1,:],rankInTimepₕ[2,:], "r:", label=L"parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚₕ)), linewidth=2, alpha=1.0)
ax.set_xlim([rankInTime[1,1],rankInTime[1,end]+0.05])
ax.set_ylim([0,max(maximum(rankInTimeₕ[2,2:end]),maximum(rankInTimepₕ[2,2:end]))+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) ;
ax.legend(loc="lower right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranks_$(s.problem)_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.NCellsX)_N$(s.nₚₙ).png")

# plot η in time
fig = figure("eta finer",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(η₁[:,1],η₁[:,2], "r-.", label=L"BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(ηp₁[:,1],ηp₁[:,2], "b--", label=L"parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(η₁Bound[:,1],η₁Bound[:,2], "r:", label=L"$c\Vert\psi\Vert\bar \vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηp₁Bound[:,1],ηp₁Bound[:,2], "b-", label=L"$c\Vert\psi\Vert\bar \vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([η₁[1,1],η₁[end,1]])
ax.set_ylim([0,max(maximum(η₁[1:end,2]),maximum(ηp₁[1:end,2]),maximum(η₁Bound[1:end,2]),maximum(ηp₁Bound[1:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_thetaBUG$(ϑᵤ)_thetaParallel$(ϑₚ)_nx$(s.Nx)_N$(s.nₚₙ).png")

# plot η in time
fig = figure("eta finest",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(η₂[:,1],η₂[:,2], "r-.", label=L"BUG, $\bar \vartheta =$ "*LaTeXString(string(ϑᵤₕ)), linewidth=2, alpha=1.0)
ax.plot(ηp₂[:,1],ηp₂[:,2], "b--", label=L"parallel, $\bar \vartheta =$ "*LaTeXString(string(ϑₚₕ)), linewidth=2, alpha=1.0)
ax.plot(η₂Bound[:,1],η₂Bound[:,2], "r:", label=L"$c\Vert\psi\Vert\bar \vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηp₂Bound[:,1],ηp₂Bound[:,2], "b-", label=L"$c\Vert\psi\Vert\bar \vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([η₂[1,1],η₂[end,1]])
ax.set_ylim([0,max(maximum(η₂[1:end,2]),maximum(ηp₂[1:end,2]),maximum(η₂Bound[1:end,2]),maximum(ηp₂Bound[1:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_thetaBUG$(ϑᵤₕ)_thetaParallel$(ϑₚₕ)_nx$(s.Nx)_N$(s.nₚₙ).png")

fig = figure("setup",figsize=(10,10),dpi=100)
ax = fig.add_subplot(111)
rect1 = matplotlib.patches.Rectangle((1,5), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,5), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,4), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,4), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,3), 1.0, 1.0, color="orange")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,3), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,3), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((4,2), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((2,2), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((3,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((1,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
rect1 = matplotlib.patches.Rectangle((5,1), 1.0, 1.0, color="black")
ax.add_patch(rect1)
ax.grid()
plt.xlim([0, 7])
plt.ylim([0, 7])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("lattice testcase", fontsize=25)
tight_layout()
plt.show()
savefig("results/setup_lattice_testcase.png")


println("main finished")
