include("settings.jl")
include("Solver.jl")

using DelimitedFiles
using NPZ
using PyPlot

s = Settings();
ϑCoarse = 2e-2;
ϑFine = 1e-2;

# run solver for different end times and different epsilons

# end time tEnd = 2, ϑ = 1e-1
s.ϑ = ϑCoarse;
s.tEnd = 2.0;
solver = Solver(s);
@time tEnd, u3P, _ = SolveParallelRejection(solver);

# end time tEnd = 2.75, ϑ = 1e-1
s.tEnd = 2.75;
solver = Solver(s);
@time tEnd, u4P, _ = SolveParallelRejection(solver);

# end time tEnd = 5, ϑ = 1e-1
s.tEnd = 5.0;
solver = Solver(s);
@time tEnd, u5P, ranks, η, ηBound = SolveParallelRejection(solver);

# end time tEnd = 2, ϑ = 1e-1
s.tEnd = 2.0;
solver = Solver(s);
@time tEnd, u3, _ = SolveBUGrank_adapt_rejection(solver);

# end time tEnd = 2.75, ϑ = 1e-1
s.tEnd = 2.75;
solver = Solver(s);
@time tEnd, u4, _ = SolveBUGrank_adapt_rejection(solver);

# end time tEnd = 5, ϑ = 1e-1
s.tEnd = 5.0;
solver = Solver(s);
@time tEnd, u5, ranksBUG, ηBug, ηBugBound = SolveBUGrank_adapt_rejection(solver);

# end time tEnd = 2, ϑ = 5e-2
s.ϑ = ϑFine;
s.tEnd = 2.0;
solver = Solver(s);
@time tEnd, u3PF, _ = SolveParallelRejection(solver);

# end time tEnd = 2.75, ϑ = 5e-2
s.tEnd = 2.75;
solver = Solver(s);
@time tEnd, u4PF, _ = SolveParallelRejection(solver);

# end time tEnd = 5, ϑ = 5e-2
s.tEnd = 5.0;
solver = Solver(s);
@time tEnd, u5PF, ranksF, ηF, ηBoundF = SolveParallelRejection(solver);

# end time tEnd = 2, ϑ = 5e-2
s.tEnd = 2.0;
solver = Solver(s);
@time tEnd, u3F, _ = SolveBUGrank_adapt_rejection(solver);

# end time tEnd = 2.75, ϑ = 5e-2
s.tEnd = 2.75;
solver = Solver(s);
@time tEnd, u4F, _ = SolveBUGrank_adapt_rejection(solver);

# end time tEnd = 5, ϑ = 5e-2
s.tEnd = 5.0;
solver = Solver(s);
@time tEnd, u5F, ranksBUGF, ηBugF, ηBoundBugF = SolveBUGrank_adapt_rejection(solver);

## plotting ##

# plot solution

## read reference solution
# t = 5
v = readdlm("PlaneSourceRawT5", ',')
uEx5 = zeros(length(v));
for i = 1:length(v)
    if v[i] == ""
        uEx5[i] = 0.0;
    else
        uEx5[i] = Float64(v[i])
    end
end
x5 = collect(range(-5,5,length=(2*length(v)-1)));
uEx5 = [uEx5[end:-1:2];uEx5];

# t = 2.75
v = readdlm("PlaneSourceRawT2-75", ',')
uEx4 = zeros(length(v));
for i = 1:length(v)
    if v[i] == ""
        uEx4[i] = 0.0;
    else
        uEx4[i] = Float64(v[i])
    end
end
x4 = collect(range(-5,5,length=(2*length(v)-1)));
uEx4 = [uEx4[end:-1:2];uEx4];

# t = 2
v = readdlm("PlaneSourceRawT2Fine", ',')
uEx3 = zeros(length(v));
for i = 1:length(v)
    if v[i] == ""
        uEx3[i] = 0.0;
    else
        uEx3[i] = Float64(v[i])
    end
end
x3 = collect(range(-5,5,length=(2*length(v)-1)));
uEx3 = [uEx3[end:-1:2];uEx3];

# start plot BUG
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(s.xMid,u3[:,1], "b--", linewidth=2, label=L"$t=2$, BUG", alpha=1.0)
ax.plot(x3,uEx3, "b-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u4[:,1], "r:", linewidth=2, label=L"$t=2.75$, BUG", alpha=1.0)
ax.plot(x4,uEx4, "r-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u5[:,1], "g-.", linewidth=2, label=L"$t=5$, BUG", alpha=1.0)
ax.plot(x5,uEx5, "g-", linewidth=2, alpha=0.5)
ylabel(L"\Phi", fontsize=30)
ax.set_xlim([-5,5])
ax.set_xlabel("x", fontsize=30);
ax.legend(loc="upper right", fontsize=30)
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/PhiBUGThetaCoarse.png")


# start plot P-BUG
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(s.xMid,u3P[:,1], "b--", linewidth=2, label=L"$t=2$, parallel", alpha=1.0)
ax.plot(x3,uEx3, "b-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u4P[:,1], "r:", linewidth=2, label=L"$t=2.75$, parallel", alpha=1.0)
ax.plot(x4,uEx4, "r-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u5P[:,1], "g-.", linewidth=2, label=L"$t=5$, parallel", alpha=1.0)
ax.plot(x5,uEx5, "g-", linewidth=2, alpha=0.5)
ylabel(L"\Phi", fontsize=30)
ax.set_xlim([-5,5])
ax.set_xlabel("x", fontsize=30);
ax.legend(loc="upper right", fontsize=30)
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/PhiParallelThetaCoarse.png")

# start plot BUG
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(s.xMid,u3F[:,1], "b--", linewidth=2, label=L"$t=2$, BUG", alpha=1.0)
ax.plot(x3,uEx3, "b-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u4F[:,1], "r:", linewidth=2, label=L"$t=2.75$, BUG", alpha=1.0)
ax.plot(x4,uEx4, "r-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u5F[:,1], "g-.", linewidth=2, label=L"$t=5$, BUG", alpha=1.0)
ax.plot(x5,uEx5, "g-", linewidth=2, alpha=0.5)
ylabel(L"\Phi", fontsize=30)
ax.set_xlim([-5,5])
ax.set_xlabel("x", fontsize=30);
ax.legend(loc="upper right", fontsize=30)
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/PhiBUGThetaFine.png")


# start plot P-BUG
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(s.xMid,u3PF[:,1], "b--", linewidth=2, label=L"$t=2$, parallel", alpha=1.0)
ax.plot(x3,uEx3, "b-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u4PF[:,1], "r:", linewidth=2, label=L"$t=2.75$, parallel", alpha=1.0)
ax.plot(x4,uEx4, "r-", linewidth=2, alpha=0.5)
ax.plot(s.xMid,u5PF[:,1], "g-.", linewidth=2, label=L"$t=5$, parallel", alpha=1.0)
ax.plot(x5,uEx5, "g-", linewidth=2, alpha=0.5)
ylabel(L"\Phi", fontsize=30)
ax.set_xlim([-5,5])
ax.set_xlabel("x", fontsize=30);
ax.legend(loc="upper right", fontsize=30)
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/PhiParallelThetaFine.png")

# plot rank in time
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(ranks[1,:],ranks[2,:], "k-", linewidth=2, alpha=1.0)
ax.plot([2,2],[0,100], "b--", linewidth=2, alpha=1.0)
ax.plot([2.75,2.75],[0,100], "r:", linewidth=2, alpha=1.0)
ax.plot([5,5],[0,100], "g-.", linewidth=2, alpha=1.0)
ax.set_xlim([ranks[1,1],ranks[1,end]+0.05])
ax.set_ylim([0,maximum(ranks[2,2:end])+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranksParallelThetaCoarse.png")

# plot rank in time
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(ranksBUG[1,:],ranksBUG[2,:], "k-", linewidth=2, alpha=1.0)
ax.plot([2,2],[0,100], "b--", linewidth=2, alpha=1.0)
ax.plot([2.75,2.75],[0,100], "r:", linewidth=2, alpha=1.0)
ax.plot([5,5],[0,100], "g-.", linewidth=2, alpha=1.0)
ax.set_xlim([ranksBUG[1,1],ranksBUG[1,end]+0.05])
ax.set_ylim([0,max(maximum(ranks[2,2:end]),maximum(ranksBUG[2,2:end]))+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranksBUGThetaCoarse.png")

# plot rank in time
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(ranksF[1,:],ranksF[2,:], "k-", linewidth=2, alpha=1.0)
ax.plot([2,2],[0,100], "b--", linewidth=2, alpha=1.0)
ax.plot([2.75,2.75],[0,100], "r:", linewidth=2, alpha=1.0)
ax.plot([5,5],[0,100], "g-.", linewidth=2, alpha=1.0)
ax.set_xlim([ranksF[1,1],ranksF[1,end]+0.05])
ax.set_ylim([0,maximum(ranksF[2,2:end])+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranksParallelThetaFine.png")

# plot rank in time
fig, ax = subplots(figsize=(15, 12), dpi=100)
ax.plot(ranksBUGF[1,:],ranksBUGF[2,:], "k-", linewidth=2, alpha=1.0)
ax.plot([2,2],[0,100], "b--", linewidth=2, alpha=1.0)
ax.plot([2.75,2.75],[0,100], "r:", linewidth=2, alpha=1.0)
ax.plot([5,5],[0,100], "g-.", linewidth=2, alpha=1.0)
ax.set_xlim([ranksBUGF[1,1],ranksBUGF[1,end]+0.05])
ax.set_ylim([0,max(maximum(ranksF[2,2:end]),maximum(ranksBUGF[2,2:end]))+2])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel("rank", fontsize=30);
ax.tick_params("both",labelsize=30) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/ranksBUGThetaFine.png")

# plot η in time
fig = figure("eta coarse",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(ηBug[:,1],ηBug[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑCoarse)), linewidth=2, alpha=1.0)
ax.plot(η[:,1],η[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑCoarse)), linewidth=2, alpha=1.0)
ax.plot(ηBugBound[:,1],ηBugBound[:,2], "r:", label=L"$c\Vert\psi\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηBound[:,1],ηBound[:,2], "b-", label=L"$c\Vert\psi\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([ηBug[1,1],ηBug[end,1]+0.05])
ax.set_ylim([0,max(maximum(ηBug[2:end,2]),maximum(η[2:end,2]),maximum(ηBound[2:end,2]),maximum(ηBugBound[2:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_theta$(ϑCoarse)_nx$(s.Nx)_N$(s.nPN).png")

# plot η fine in time
fig = figure("eta fine",figsize=(15,12),dpi=100)
ax = gca()
ax.plot(ηBugF[:,1],ηBugF[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑFine)), linewidth=2, alpha=1.0)
ax.plot(ηF[:,1],ηF[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑFine)), linewidth=2, alpha=1.0)
ax.plot(ηBoundBugF[:,1],ηBoundBugF[:,2], "r:", label=L"$c\Vert\psi\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(ηBoundF[:,1],ηBoundF[:,2], "b-", label=L"$c\Vert\psi\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([ηBugF[1,1],ηBugF[end,1]+0.05])
ax.set_ylim([0,max(maximum(ηBugF[2:end,2]),maximum(ηF[2:end,2]),maximum(ηBoundF[2:end,2]),maximum(ηBoundBugF[2:end,2]))+1])
ax.set_xlabel("time", fontsize=30);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/eta_planesource_theta$(ϑFine)_nx$(s.Nx)_N$(s.nPN).png")

println("main finished")
