__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum

include("PNSystem.jl")

struct SolverDLRA
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1};
    # Roe matrix
    AbsAx::Array{Float64,2};
    AbsAz::Array{Float64,2};

    # functionalities of the PN system
    pn::PNSystem;

    Dxx::SparseMatrixCSC{Float64, Int64};
    Dyy::SparseMatrixCSC{Float64, Int64};
    Dx::SparseMatrixCSC{Float64, Int64};
    Dy::SparseMatrixCSC{Float64, Int64};

    # constructor
    function SolverDLRA(settings)
        x = settings.x;
        y = settings.y;

        # setup flux matrix
        γ = zeros(settings.nₚₙ+1);
        for i = 1:settings.nₚₙ+1
            n = i-1;
            γ[i] = 2/(2*n+1);
        end

        # construct PN system matrices
        pn = PNSystem(settings)
        SetupSystemMatrices(pn);

        # setup Roe matrix
        S = eigvals(pn.Ax)
        V = eigvecs(pn.Ax)
        AbsAx = V*abs.(diagm(S))*inv(V)

        S = eigvals(pn.Az)
        V = eigvecs(pn.Az)
        AbsAz = V*abs.(diagm(S))*inv(V)

        # setupt stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        Dxx = spzeros(nx*ny,nx*ny);
        Dyy = spzeros(nx*ny,nx*ny);
        Dx = spzeros(nx*ny,nx*ny);
        Dy = spzeros(nx*ny,nx*ny);

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3*(nx-2)*(ny-2)); J = zeros(3*(nx-2)*(ny-2)); vals = zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # x part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.Δx; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.Δx;
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.Δx; 
                end
            end
        end
        Dxx = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(3*(nx-2)*(ny-2)); J .= zeros(3*(nx-2)*(ny-2)); vals .= zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # y part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.Δy; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.Δy;
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.Δy; 
                end
            end
        end
        Dyy = sparse(II,J,vals,nx*ny,nx*ny);

        II = zeros(2*(nx-2)*(ny-2)); J = zeros(2*(nx-2)*(ny-2)); vals = zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # x part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.Δx;
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.Δx;
                end
            end
        end
        Dx = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(2*(nx-2)*(ny-2)); J .= zeros(2*(nx-2)*(ny-2)); vals .= zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # y part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.Δy;
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.Δy;
                end
            end
        end
        Dy = sparse(II,J,vals,nx*ny,nx*ny);

        new(x,y,settings,γ,AbsAx,AbsAz,pn,Dxx,Dyy,Dx,Dy);
    end
end

function SetupIC(obj::SolverDLRA)
    u = zeros(obj.settings.NCellsX,obj.settings.NCellsY,obj.pn.nTotalEntries);
    u[:,:,1] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)
    return u;
end

function Solve(obj::SolverDLRA)
    # Get rank
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0;

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 

    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    t = 0.0;

    for n in 1:nT

        u .=  u .- Δt*(obj.Dx*u*obj.pn.Ax .+ obj.Dy*u*obj.pn.Az .+ obj.Dxx*u*obj.AbsAx .+ obj.Dyy*u*obj.AbsAz .+ σₜ*u .- σₛ*u*E₁ .- Q*e₁')

        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*u[:,1];

end

function SolveBUG(obj::SolverDLRA)
    # Get rank
    r=50;
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0;

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    prog = Progress(nT,1)

    # Low-rank approx of init data:
    X,S,W = svd(u);

    # free memory
    u = 0;
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Mᵤ = zeros(r,r)
    Nᵤ = zeros(r,r)

    X₁ = zeros(nx*ny,r)

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 
    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    t = 0.0;

    for n=1:nT

        ################## K-step ##################
        K = X*S;

        WᵀAₓ₂W = W'*obj.pn.Az*W
        WᵀRₓ₂W = W'*obj.AbsAz*W
        WᵀRₓ₁W = W'*obj.AbsAx*W
        WᵀAₓ₁W = W'*obj.pn.Ax*W
        WᵀE₁W = W'*E₁*W;

        K = K .- Δt*(obj.Dx*K*WᵀAₓ₁W .+ obj.Dy*K*WᵀAₓ₂W .+ obj.Dxx*K*WᵀRₓ₁W .+ obj.Dyy*K*WᵀRₓ₂W .+ σₜ*K .- σₛ*K*WᵀE₁W .- Q*(e₁'*W));

        X₁,_ = qr!(K);
        X₁ = Matrix(X₁)
        X₁ = X₁[:,1:r];

        Mᵤ = X₁' * X;
        ################## L-step ##################
        L = W*S';

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        L .= L .- Δt*(obj.pn.Ax*L*XᵀDₓ₁X' .+ obj.pn.Az*L*XᵀDₓ₂X' .+ obj.AbsAx*L*XᵀDxxX' .+ obj.AbsAz*L*XᵀDyyX' .+ L*XᵀσₜX .- E₁*L*XᵀσₛX .- e₁*(Q'*X));
                
        W₁,_ = qr(L);
        W₁ = Matrix(W₁)
        W₁ = W₁[:,1:r];

        Nᵤ = W₁' * W;
        W = W₁;
        X = X₁;
        ################## S-step ##################
        S = Mᵤ*S*(Nᵤ')

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X

        WᵀAₓ₂W = W'*obj.pn.Az*W
        WᵀRₓ₂W = W'*obj.AbsAz*W
        WᵀRₓ₁W = W'*obj.AbsAx*W
        WᵀAₓ₁W = W'*obj.pn.Ax*W
        WᵀE₁W = W'*E₁*W;
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        S .= S .- Δt .* (XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S .- XᵀσₛX*S*WᵀE₁W .- (X'*Q)*(e₁'*W)));
        
        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*X*S*W[1,:];

end

function SolveBUGAdaptive(obj::SolverDLRA)
    # Get rank
    r=obj.settings.rMax;
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0;

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X,S,W = svd(u);
    # free memory
    u = 0;
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 

    rankInTime = zeros(2,nT);
    NormInTime = zeros(2,nT);

    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    t = 0.0;

    for n=1:nT
        rankInTime[1,n] = t;
        rankInTime[2,n] = r;
        NormInTime[1,n] = t;
        NormInTime[2,n] = norm(S,2);

        ################## K-step ##################
        K = X*S;

        WᵀAₓ₂W = W'*obj.pn.Az*W
        WᵀRₓ₂W = W'*obj.AbsAz*W
        WᵀRₓ₁W = W'*obj.AbsAx*W
        WᵀAₓ₁W = W'*obj.pn.Ax*W
        WᵀE₁W = W'*E₁*W;

        K = K .- Δt*(obj.Dx*K*WᵀAₓ₁W .+ obj.Dy*K*WᵀAₓ₂W .+ obj.Dxx*K*WᵀRₓ₁W .+ obj.Dyy*K*WᵀRₓ₂W .+ σₜ*K .- σₛ*K*WᵀE₁W .- Q*(e₁'*W));

        K = [K X];
        X₁,_ = qr!(K);
        X₁ = Matrix(X₁)
        X₁ = X₁[:,1:2*r];

        Mᵤ = X₁' * X;
        ################## L-step ##################
        L = W*S';

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        L .= L .- Δt*(obj.pn.Ax*L*XᵀDₓ₁X' .+ obj.pn.Az*L*XᵀDₓ₂X' .+ obj.AbsAx*L*XᵀDxxX' .+ obj.AbsAz*L*XᵀDyyX' .+ L*XᵀσₜX .- E₁*L*XᵀσₛX .- e₁*(Q'*X));
                
        L = [L W];
        W₁,_ = qr(L);
        W₁ = Matrix(W₁)
        W₁ = W₁[:,1:2*r];

        Nᵤ = W₁' * W;
        W = W₁;
        X = X₁;
        ################## S-step ##################
        S = Mᵤ*S*(Nᵤ')

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X

        WᵀAₓ₂W = W'*obj.pn.Az'*W
        WᵀRₓ₂W = W'*obj.AbsAz'*W
        WᵀRₓ₁W = W'*obj.AbsAx'*W
        WᵀAₓ₁W = W'*obj.pn.Ax'*W
        WᵀE₁W = W'*E₁*W;
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        S .= S .- Δt .* (XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S .- XᵀσₛX*S*WᵀE₁W .- (X'*Q)*(e₁'*W)));
        
        ################## truncate ##################

        X, S, W = truncate!(obj, X, S, W)

        # update rank
        r = size(S,1);
        
        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*X*S*W[1,:],rankInTime;

end

function SolveBUGAdaptiveRejection(obj::SolverDLRA)
    # Get rank
    rmax = obj.settings.rMax;
    r=obj.settings.rMax;
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0;

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X,S,W = svd(u);
    # free memory
    u = 0;
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 

    timeVec = [];
    rankInTime = [];
    etaVec = [];
    etaVecTime = [];
    etaBoundVec = [];

    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    t = 0.0;
    n = 0;

    while t < nT*Δt
        n = n+1;
        timeVec = [timeVec; t];
        rankInTime = [rankInTime; r];

        S0 = S;

        ################## K-step ##################
        K = X*S;

        WᵀAₓ₂W = W'*obj.pn.Az*W
        WᵀRₓ₂W = W'*obj.AbsAz*W
        WᵀRₓ₁W = W'*obj.AbsAx*W
        WᵀAₓ₁W = W'*obj.pn.Ax*W
        WᵀE₁W = W'*E₁*W;

        K = K .- Δt*(obj.Dx*K*WᵀAₓ₁W .+ obj.Dy*K*WᵀAₓ₂W .+ obj.Dxx*K*WᵀRₓ₁W .+ obj.Dyy*K*WᵀRₓ₂W .+ σₜ*K .- σₛ*K*WᵀE₁W .- Q*(e₁'*W));

        X₁,_ = qr([X K]);
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(r+1):(2*r)];

        Mᵤ = X₁' * X;
        ################## L-step ##################
        L = W*S';

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        L .= L .- Δt*(obj.pn.Ax*L*XᵀDₓ₁X' .+ obj.pn.Az*L*XᵀDₓ₂X' .+ obj.AbsAx*L*XᵀDxxX' .+ obj.AbsAz*L*XᵀDyyX' .+ L*XᵀσₜX .- E₁*L*XᵀσₛX .- e₁*(Q'*X));

        W₁,_ = qr([W L]);
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(r+1):(2*r)];

        Nᵤ = W₁' * W;
        W = W₁;
        X = X₁;
        ################## S-step ##################
        S = Mᵤ*S*(Nᵤ')

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X

        WᵀAₓ₂W = W'*obj.pn.Az'*W
        WᵀRₓ₂W = W'*obj.AbsAz'*W
        WᵀRₓ₁W = W'*obj.AbsAx'*W
        WᵀAₓ₁W = W'*obj.pn.Ax'*W
        WᵀE₁W = W'*E₁*W;
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        Sup = S .- Δt .* (XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S .- XᵀσₛX*S*WᵀE₁W .- (X'*Q)*(e₁'*W)));

        ################## truncate ##################

        XUP, SUP, WUP = truncate!(obj, X, Sup, W)
        
        # rejection step
        if size(SUP,1) == 2*r && 2*r < rmax
            r = 2*r;
            n = n-1;
            continue;
        else
            XᵀDₓ₁X = tildeX₁'*obj.Dx*X
            XᵀDₓ₂X = tildeX₁'*obj.Dy*X
            XᵀDxxX = tildeX₁'*obj.Dxx*X
            XᵀDyyX = tildeX₁'*obj.Dyy*X

            WᵀAₓ₂W = W'*obj.pn.Az'*tildeW₁
            WᵀRₓ₂W = W'*obj.AbsAz'*tildeW₁
            WᵀRₓ₁W = W'*obj.AbsAx'*tildeW₁
            WᵀAₓ₁W = W'*obj.pn.Ax'*tildeW₁
            WᵀE₁W = W'*E₁*tildeW₁;
            XᵀσₛX = tildeX₁'*σₛ*X
            XᵀσₜX = tildeX₁'*σₜ*X

            eta = norm(XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S*(W'*tildeW₁) .- XᵀσₛX*S*WᵀE₁W .- (tildeX₁'*Q)*(e₁'*tildeW₁)))

            etaVec = [etaVec; eta];
            etaVecTime = [etaVecTime; t];
            bound = obj.settings.cη * obj.settings.ϑ * max(1e-7,norm(Sup)^obj.settings.ϑIndex) / Δt
            etaBoundVec = [etaBoundVec; bound]

            if eta > bound && 2*r < rmax
                println(eta," > ",obj.settings.cη * obj.settings.ϑ * max(1e-7,norm(Sup)^obj.settings.ϑIndex) / Δt)
                r = 2*r;
                n = n-1;
                continue;
            end
        end

        # update rank
        S = SUP;
        X = XUP;
        W = WUP;
        r = size(S,1);
        
        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*X*S*W[1,:],[timeVec rankInTime]', [etaVecTime etaVec], [etaVecTime etaBoundVec];;

end

function SolveParallel(obj::SolverDLRA)
    # Get rank
    r=obj.settings.rMax;
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0; 

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X,S,W = svd(u);

    # free memory
    u = 0; 
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 

    rankInTime = zeros(2,nT);
    NormInTime = zeros(2,nT);

    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    t = 0.0;

    for n=1:nT
        rankInTime[1,n] = t;
        rankInTime[2,n] = r;
        NormInTime[1,n] = t;
        NormInTime[2,n] = norm(S,2);

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X

        WᵀAₓ₂W = W'*obj.pn.Az'*W
        WᵀRₓ₂W = W'*obj.AbsAz'*W
        WᵀRₓ₁W = W'*obj.AbsAx'*W
        WᵀAₓ₁W = W'*obj.pn.Ax'*W
        WᵀE₁W = W'*E₁*W;
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        ################## K-step ##################
        K = X*S;

        K = K .- Δt*(obj.Dx*K*WᵀAₓ₁W .+ obj.Dy*K*WᵀAₓ₂W .+ obj.Dxx*K*WᵀRₓ₁W .+ obj.Dyy*K*WᵀRₓ₂W .+ σₜ*K .- σₛ*K*WᵀE₁W .- Q*(e₁'*W));

        X₁,_ = qr([X K]);
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(r+1):(2*r)];

        ################## L-step ##################
        L = W*S';

        L .= L .- Δt*(obj.pn.Ax*L*XᵀDₓ₁X' .+ obj.pn.Az*L*XᵀDₓ₂X' .+ obj.AbsAx*L*XᵀDxxX' .+ obj.AbsAz*L*XᵀDyyX' .+ L*XᵀσₜX .- E₁*L*XᵀσₛX .- e₁*(Q'*X));
                
        W₁,_ = qr([W L]);
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(r+1):(2*r)];

        ################## S-step ##################

        S .= S .- Δt .* (XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S .- XᵀσₛX*S*WᵀE₁W .- (X'*Q)*(e₁'*W)));
        
        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = S;
        SNew[(r+1):end,1:r] = tildeX₁'*K;
        SNew[1:r,(r+1):(2 * r)] = L' * tildeW₁;

        # truncate
        X, S, W = truncate!(obj,[X tildeX₁],SNew,[W tildeW₁]);

        # update rank
        r = size(S,1);
        
        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*X*S*W[1,:],rankInTime;

end

function SolveParallelRejection(obj::SolverDLRA)
    # Get rank
    rmax = obj.settings.rMax;
    r=obj.settings.rMax;
    s = obj.settings;
    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end
    # free memory
    v = 0; 

    nT = Int(ceil(s.tₑ/s.Δt))
    Δt = s.Δt

    # Low-rank approx of init data:
    X,S,W = svd(u);

    # free memory
    u = 0; 
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];

    e₁ = zeros(N); e₁[1] = 1.0; 
    E₁ = sparse([1],[1],[1.0],N,N); 

    rankInTime = [];
    timeVec = [];
    etaVec = [];
    etaVecTime = [];
    etaBoundVec = [];

    σₛ = Diagonal(obj.settings.σₛ);
    σₜ = Diagonal(obj.settings.σₐ .+ obj.settings.σₛ);
    Q = obj.settings.Q;

    prog = Progress(nT,1)
    
    t = 0.0;
    n = 0;

    while t < nT*Δt
        n += 1;
        
        timeVec = [timeVec; t];
        rankInTime = [rankInTime; r];

        XᵀDₓ₁X = X'*obj.Dx*X
        XᵀDₓ₂X = X'*obj.Dy*X
        XᵀDxxX = X'*obj.Dxx*X
        XᵀDyyX = X'*obj.Dyy*X

        WᵀAₓ₂W = W'*obj.pn.Az'*W
        WᵀRₓ₂W = W'*obj.AbsAz'*W
        WᵀRₓ₁W = W'*obj.AbsAx'*W
        WᵀAₓ₁W = W'*obj.pn.Ax'*W
        WᵀE₁W = W'*E₁*W;
        XᵀσₛX = X'*σₛ*X
        XᵀσₜX = X'*σₜ*X

        ################## K-step ##################
        K = X*S;

        K = K .- Δt*(obj.Dx*K*WᵀAₓ₁W .+ obj.Dy*K*WᵀAₓ₂W .+ obj.Dxx*K*WᵀRₓ₁W .+ obj.Dyy*K*WᵀRₓ₂W .+ σₜ*K .- σₛ*K*WᵀE₁W .- Q*(e₁'*W));

        X₁,_ = qr([X K]);
        X₁ = Matrix(X₁)
        tildeX₁ = X₁[:,(r+1):end];

        ################## L-step ##################
        L = W*S';

        L .= L .- Δt*(obj.pn.Ax*L*XᵀDₓ₁X' .+ obj.pn.Az*L*XᵀDₓ₂X' .+ obj.AbsAx*L*XᵀDxxX' .+ obj.AbsAz*L*XᵀDyyX' .+ L*XᵀσₜX .- E₁*L*XᵀσₛX .- e₁*(Q'*X));
                
        W₁,_ = qr([W L]);
        W₁ = Matrix(W₁)
        tildeW₁ = W₁[:,(r+1):end];

        ################## S-step ##################

        Sup = S .- Δt .* (XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S .- XᵀσₛX*S*WᵀE₁W .- (X'*Q)*(e₁'*W)));
        
        ################## truncate ##################

        SNew = zeros(2 * r, 2 * r);

        SNew[1:r,1:r] = Sup;
        SNew[(r+1):end,1:r] = tildeX₁'*K;
        SNew[1:r,(r+1):(2 * r)] = L' * tildeW₁;

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
            XᵀDₓ₁X = tildeX₁'*obj.Dx*X
            XᵀDₓ₂X = tildeX₁'*obj.Dy*X
            XᵀDxxX = tildeX₁'*obj.Dxx*X
            XᵀDyyX = tildeX₁'*obj.Dyy*X
    
            WᵀAₓ₂W = W'*obj.pn.Az'*tildeW₁
            WᵀRₓ₂W = W'*obj.AbsAz'*tildeW₁
            WᵀRₓ₁W = W'*obj.AbsAx'*tildeW₁
            WᵀAₓ₁W = W'*obj.pn.Ax'*tildeW₁
            WᵀE₁W = W'*E₁*tildeW₁;
            XᵀσₛX = tildeX₁'*σₛ*X
            XᵀσₜX = tildeX₁'*σₜ*X

            eta = norm(XᵀDₓ₁X*S*WᵀAₓ₁W .+ XᵀDₓ₂X*S*WᵀAₓ₂W .+ XᵀDxxX*S*WᵀRₓ₁W .+ XᵀDyyX*S*WᵀRₓ₂W .+ (XᵀσₜX*S*(W'*tildeW₁) .- XᵀσₛX*S*WᵀE₁W .- (tildeX₁'*Q)*(e₁'*tildeW₁)))

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

        # update rank
        S = SUP;
        X = XUP;
        W = WUP;
        r = size(S,1);
        
        t += Δt;
        next!(prog) # update progress bar
    end

    # return end time and solution
    return 0.5*sqrt(obj.γ[1])*X*S*W[1,:],[timeVec rankInTime]', [etaVecTime etaVec], [etaVecTime etaBoundVec];

end

function truncate!(obj::SolverDLRA,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    ϑ = obj.settings.ϑ*max(1e-11,norm(D)^obj.settings.ϑIndex);
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < ϑ
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
