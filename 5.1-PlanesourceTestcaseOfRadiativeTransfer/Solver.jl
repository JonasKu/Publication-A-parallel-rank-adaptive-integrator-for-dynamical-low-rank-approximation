__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using PyCall

struct Solver
    # spatial grid of cell interfaces
    x::Array{Float64};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};

    # stencil matrices
    Dₓ::Tridiagonal{Float64, Vector{Float64}};
    Dₓₓ::Tridiagonal{Float64, Vector{Float64}};

    # physical parameters
    σₐ::Float64;
    σₛ::Float64;

    # constructor
    function Solver(settings)
        x = settings.x;
        nx = settings.NCells;
        Δx = settings.Δx;

        # setup flux matrix
        γ = ones(settings.nPN);

        # setup γ vector
        γ = zeros(settings.nPN);
        for i = 1:settings.nPN
            n = i-1;
            γ[i] = 2/(2*n+1);
        end

        # setup PN system matrix
        upper = [(n+1)/(2*n+1)*sqrt(γ[n+2])/sqrt(γ[n+1]) for n = 0:settings.nPN-2];
        lower = [n/(2*n+1)*sqrt(γ[n])/sqrt(γ[n+1]) for n = 1:settings.nPN-1];
        A = Tridiagonal(lower,zeros(settings.nPN),upper)
        
        # setup Roe matrix
        S = eigvals(Matrix(A))
        V = eigvecs(Matrix(A))
        AbsA = V*abs.(Diagonal(S))*inv(V)

        # set up spatial stencil matrices
        Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
        Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

        new(x,settings,γ,A,AbsA,Dₓ,Dₓₓ,settings.σₐ,settings.σₛ);
    end
end

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.γ[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

# full PN method
function Solve(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = SetupIC(obj);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ= Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ= Diagonal(ones(nx)).*obj.settings.σₐ;
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        u = u .- Δt * obj.Dₓ*u*A' .+ Δt * obj.Dₓₓ*u*AbsA' .- Δt * σₐ*u .- Δt * σₛ*u*G; 
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*u;

end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

# projector splitting integrator (stabilized)
function SolvePSI(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,tildeS1,tildeS2 = svd!(K); tildeS = Diagonal(tildeS1)*tildeS2'; # use svd instead of QR since svd is more efficient in Julia

        # S-step
        XDₓₓX = X1'*obj.Dₓₓ*X1
        XDₓX = X1'*obj.Dₓ*X1
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        S = tildeS .+ Δt * XDₓX*tildeS*WAW .+ Δt * XDₓₓX*tildeS*WAbsAW .+ Δt * XσₐX*tildeS .+ Δt * XσₛX*tildeS*WGW;

        # L-step
        L = W*S';
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; 
        W1,S2,S1 = svd!(L); S .= (Diagonal(S2)*S1')'; # use svd instead of QR since svd is more efficient in Julia

        X .= X1; W .= W1;
        
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W';

end

# unconventional integrator
function SolveBUG(obj::Solver)
    # store values from settings into function
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r; # rank

    # time settings
    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    # spatial and moment settings
    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # compute scattering interaction cross-sections
    G = Diagonal([0.0;ones(N-1)]); # scattering diagonal
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ;   # scattering
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;   # absorption

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        # K-step
        K = X*S; 
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;

        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = py"qr"(K);
        M = X1'*X;

        # L-step
        L = W*S';
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        W1,_ = py"qr"(L); 
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        XDₓX = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;

        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        X .= X1; W .= W1;
        
        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W';

end

# new parallel integrator
function SolveParallel(obj::Solver)
    # store values from settings into function
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r; # rank

    # time settings
    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    # spatial and moment settings
    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # compute scattering interaction cross-sections
    G = Diagonal([0.0;ones(N-1)]); # scattering diagonal
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ;   # scattering
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;   # absorption

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    rVec = zeros(2,nt)

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        r = size(S,1);

        # K-step (parallel)
        K = X*S; 
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;

        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K

        Xtmp,_ = py"qr"([X K]); X1Tilde = Xtmp[:,(r+1):end];

        # L-step (parallel)
        L = W*S';
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L

        Wtmp,_ = py"qr"([W L]); W1Tilde = Wtmp[:,(r+1):end];

        # S-step (parallel)
        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = S;
        SNew[(r+1):end,1:r] = X1Tilde'*K;
        SNew[1:r,(r+1):end] = L' * W1Tilde;

        # truncate
        X, S, W = truncate!(obj,[X X1Tilde],SNew,[W W1Tilde]);
        rVec[1,n] = t;
        rVec[2,n] = r;

        t += Δt;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W', rVec;

end

# new parallel integrator
function SolveParallelRejection(obj::Solver)
    # store values from settings into function
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r; # rank

    # time settings
    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    # spatial and moment settings
    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    # compute scattering interaction cross-sections
    G = Diagonal([0.0;ones(N-1)]); # scattering diagonal
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ;   # scattering
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;   # absorption

    etaVec = [];
    etaVecTime = [];
    etaBoundVec = [];
    timeVec = [];
    rankInTime = [];

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)
    #loop over time
    t = 0.0;
    n = 0;

    while t < nt*Δt
        n += 1;

        timeVec = [timeVec; t];
        rankInTime = [rankInTime; r];

        r = size(S,1);

        # K-step (parallel)
        K = X*S; 
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;

        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K

        Xtmp,_ = qr([X K]); tildeX₁ = Matrix(Xtmp); tildeX₁ = tildeX₁[:,(r+1):end];

        # L-step (parallel)
        L = W*S';
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L

        Wtmp,_ = qr([W L]); tildeW₁ = Matrix(Wtmp); tildeW₁ = tildeW₁[:,(r+1):end];

        # S-step (parallel)
        SBar = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = SBar;
        SNew[(r+1):end,1:r] = tildeX₁'*K;
        SNew[1:r,(r+1):end] = L' * tildeW₁;

        # truncate
        XUP, SUP, WUP = truncate!(obj,[X tildeX₁],SNew,[W tildeW₁]);

        # rejection step
        if size(SUP,1) == 2*r && 2*r < rmax
            S = ([X tildeX₁]'*X)*S*(W'*[W tildeW₁])
            X = [X tildeX₁];
            W = [W tildeW₁];
            r = 2*r;
            n = n-1;
            continue;
        else
            XDₓₓX = tildeX₁'*obj.Dₓₓ*X;
            XDₓX = tildeX₁'*obj.Dₓ*X;
            XσₐX = tildeX₁'*σₐ*X;
            XσₛX = tildeX₁'*σₛ*X;

            WAW = W'*A'*tildeW₁;
            WAbsAW = W'*AbsA'*tildeW₁;
            WGW = W'*G*tildeW₁;

            eta = norm(XDₓX*S*WAW .+ XDₓₓX*S*WAbsAW .- XσₐX*S .- XσₛX*S*WGW)

            etaVec = [etaVec; eta]
            etaVecTime = [etaVecTime; t]
            bound = obj.settings.cη * obj.settings.ϑ * max(1e-11,norm(SNew)^obj.settings.ϑIndex) / Δt;
            etaBoundVec = [etaBoundVec; bound]

            if eta > bound && 2*r < rmax
                println(eta," > ",obj.settings.cη * obj.settings.ϑ * max(1e-7,norm(SNew)^obj.settings.ϑIndex) / Δt)
                S = ([X tildeX₁]'*X)*S*(W'*[W tildeW₁])
                X = [X tildeX₁];
                W = [W tildeW₁];
                r = 2*r;
                n = n-1;
                continue;
            end
        end

        X = XUP;
        S = SUP;
        W = WUP;

        t += Δt;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',[timeVec rankInTime]', [etaVecTime etaVec], [etaVecTime etaBoundVec];

end

# rank adaptive BUG integrator
function SolveBUGrank_adapt(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    rVec = zeros(2,nt)

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        r = size(S,1);

        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = py"qr"([K X]); # use svd instead of QR since svd is more efficient in Julia
        M = X1'*X;

        # L-step
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L = W*S';
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        W1,_ = py"qr"([L W]); # use svd instead of QR since svd is more efficient in Julia
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        XDₓX = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;
        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        # truncate
        X, S, W = truncate!(obj,X1,S,W1);

        rVec[1,n] = t;
        rVec[2,n] = r;

        t += Δt;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',rVec;
end

# rank adaptive BUG integrator
function SolveBUGrank_adapt_rejection(obj::Solver)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = SetupIC(obj);

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;

    etaVec = [];
    etaBoundVec = [];
    etaVecTime = [];
    timeVec = [];
    rankInTime = [];

    # flux matrix and Roe matrix
    A = obj.A;
    AbsA = obj.AbsA;

    prog = Progress(nt,1)

    #loop over time
    t = 0.0;
    n = 0;
    while t < nt*Δt
        n += 1;

        timeVec = [timeVec; t];
        rankInTime = [rankInTime; r];

        r = size(S,1);

        # K-step
        K = X*S;
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * obj.Dₓ*K*WAW .+ Δt * obj.Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        X1,_ = qr([X K]);
        X1 = Matrix(X1)
        tildeX₁ = X1[:,(r+1):(2*r)];
        M = X1'*X;

        # L-step
        XDₓₓX = X'*obj.Dₓₓ*X;
        XDₓX = X'*obj.Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L = W*S';
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L

        W1,_ = qr([W L]);
        W1 = Matrix(W1)
        tildeW₁ = W1[:,(r+1):(2*r)];

        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*obj.Dₓₓ*X1;
        XDₓX = X1'*obj.Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;
        SNew = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        # truncate
        XUP, SUP, WUP = truncate!(obj,X1,SNew,W1);

        # rejection step
        # rejection step
        if size(SUP,1) == 2*r && 2*r < rmax
            X = X1;
            W = W1;
            r = 2*r;
            n = n-1;
            continue;
        else
            XDₓₓX = tildeX₁'*obj.Dₓₓ*X1;
            XDₓX = tildeX₁'*obj.Dₓ*X1;
            XσₐX = tildeX₁'*σₐ*X1;
            XσₛX = tildeX₁'*σₛ*X1;

            WAW = W1'*A'*tildeW₁;
            WAbsAW = W1'*AbsA'*tildeW₁;
            WGW = W1'*G*tildeW₁;

            eta = norm(XDₓX*S*WAW .+ XDₓₓX*S*WAbsAW .- XσₐX*S*(W1'*tildeW₁) .- XσₛX*S*WGW)

            etaVec = [etaVec; eta]
            etaVecTime = [etaVecTime; t]
            bound = obj.settings.cη * obj.settings.ϑ * max(1e-11,norm(SNew)^obj.settings.ϑIndex) / Δt
            etaBoundVec = [etaBoundVec; bound]

            if eta > bound && 2*r < rmax
                println(eta," > ",obj.settings.cη * obj.settings.ϑ * max(1e-7,norm(SNew)^obj.settings.ϑIndex) / Δt)
                X = X1;
                W = W1;
                r = 2*r;
                n = n-1;
                continue;
            end
        end

        X = XUP;
        S = SUP;
        W = WUP;

        t += Δt;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*X*S*W',[timeVec rankInTime]', [etaVecTime etaVec], [etaVecTime etaBoundVec];
end

function truncate!(obj::Solver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    tol = obj.settings.ϑ*max(1e-11,norm(D)^obj.settings.ϑIndex);

    rmax = Int(floor(size(D,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < tol
            rmax = j;
            break;
        end
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rMinTotal);

    # return rank
    return X*U[:, 1:rmax], diagm(D[1:rmax]), W*V[:, 1:rmax];
end

function truncateToR!(obj::Solver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    r = obj.settings.r;
   
    S = diagm(D[1:r])

    # update solution with new rank
    X = X*U[:,1:r];
    W = W*V[:,1:r];
    return X,S,W
end
