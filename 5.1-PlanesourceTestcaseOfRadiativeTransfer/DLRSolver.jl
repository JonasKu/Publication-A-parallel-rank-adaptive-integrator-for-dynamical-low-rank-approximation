__precompile__
include("quadrature.jl")
include("basis.jl")

using ProgressMeter
using LinearAlgebra
using PyCall

struct DLRSolver
    # spatial grid of cell interfaces
    x::Array{Float64,1};

    # Solver settings
    settings::Settings;

    # preallocate memory for performance
    outRhs::Array{Float64,2};
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;

    # constructor
    function DLRSolver(settings)
        x = settings.x;

        outRhs = zeros(settings.NCells,settings.nPN);

        # setup gamma vector
        gamma = zeros(settings.nPN);
        for i = 1:settings.nPN
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        
        # setup flux matrix
        A = zeros(settings.nPN,settings.nPN)
        if settings.problem == "UQ" # UQ for advection equation
            N = settings.nPN;
            q = Quadrature(2*N,"Gauss");
            b = Basis(q,settings);
            aXi = (1.0 - settings.sigmaS) .+settings.sigmaS*q.xi;
            for i = 1:N
                for j = 1:N
                    A[i,j] = IntegralVec(q, aXi.*b.PhiQuad[:,i].*b.PhiQuad[:,j]*0.5,-1.0,1.0)
                end
            end
        else # radiative transfer
            for i = 1:(settings.nPN-1)
                n = i-1;
                A[i,i+1] = (n+1)/(2*n+1)*sqrt(gamma[i+1])/sqrt(gamma[i]);
            end

            for i = 2:settings.nPN
                n = i-1;
                A[i,i-1] = n/(2*n+1)*sqrt(gamma[i-1])/sqrt(gamma[i]);
            end
        end

        # update dt with correct maximal speed lmax
        lmax = maximum(abs.(eigvals(A)));
        settings.dt = settings.dx*settings.cfl/lmax;

        new(x,settings,outRhs,gamma,A,settings.sigmaT,settings.sigmaS);
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""


function SetupIC(obj::DLRSolver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.gamma[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

function SolvePBUG(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    dx = obj.x[2]-obj.x[1];
    Nx = obj.settings.NCells;
    r = obj.settings.r; # DLR rank
    N = obj.settings.nPN; # here, N is the number of quadrature points

    r = obj.settings.r;

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = diagm(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    L = zeros(N,r);

    c1 = zeros(r,r);
    c2 = zeros(r,r);
    
    Nt = Integer(round(tEnd/dt));

    prog = Progress(Nt,1)
    for n = 1:Nt

        ############## Streaming ##############

        ###### K-step ######
        K = X*S;

        y = zeros(Nx,r);
        WAW = W'*obj.A*W;
        for j = 2:(obj.settings.NCells-1) # leave out ghost cells
            y[j,:] = (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2- (WAW*(K[j+1,:]-K[j-1,:])/dx/2);
            # add source term and scattering
            y[j,:] += - obj.sigmaT*K[j,:];
            y[j,:] += (obj.sigmaS*K[j,:]*W[1,:]')'*W[1,:]
        end

        K .= K + dt*y;

        X1,SK = py"qr"(K); M = X1'*X;

        ###### L-step ######
        L = W*S';

        c1 .= zeros(r,r);
        c2 .= zeros(r,r);
        
        for k = 1:r
            for l = 1:r
                for j = 2:(obj.settings.NCells-1)
                    c1[k,l] = c1[k,l] + X[j,k].*(X[j+1,l]-2*X[j,l]+X[j-1,l])/dt/2;
                    c2[k,l] = c2[k,l] + X[j,k].*(X[j+1,l]-X[j-1,l])/dx/2;
                end
            end
        end

        y = L*c1' - obj.A*L*c2' - obj.sigmaT*L;
        y[1,:] += obj.sigmaS*L[1,:]

        L .= L .+ dt*y;

        W1,SL = py"qr"(L); N = W1'*W; SL = SL';

        # S-step
        W .= W1; X .= X1;
        S = 0.5 * (SK*N' + M*SL);
      
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t,0.5*sqrt(obj.gamma[1])*X*S*W';

end

function SolveChristian(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    dx = obj.x[2]-obj.x[1];
    Nx = obj.settings.NCells;
    r = obj.settings.r; # DLR rank
    N = obj.settings.nPN; # here, N is the number of quadrature points

    r = obj.settings.r;

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = diagm(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    L = zeros(N,r);

    c1 = zeros(r,r);
    c2 = zeros(r,r);
    
    Nt = Integer(round(tEnd/dt));

    prog = Progress(Nt,1)
    for n = 1:Nt

        ############## Streaming ##############

        ###### K-step ######
        K = X*S;

        y = zeros(Nx,r);
        WAW = W'*obj.A*W;
        for j = 2:(obj.settings.NCells-1) # leave out ghost cells
            y[j,:] = (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2- (WAW*(K[j+1,:]-K[j-1,:])/dx/2);
            # add source term and scattering
            y[j,:] += - obj.sigmaT*K[j,:];
            y[j,:] += (obj.sigmaS*K[j,:]*W[1,:]')'*W[1,:]
        end

        K .= K + dt*y;

        X1,SK = py"qr"(K); M = X1'*X;

        ###### L-step ######
        L = W*S';

        c1 .= zeros(r,r);
        c2 .= zeros(r,r);
        
        for k = 1:r
            for l = 1:r
                for j = 2:(obj.settings.NCells-1)
                    c1[k,l] = c1[k,l] + X[j,k].*(X[j+1,l]-2*X[j,l]+X[j-1,l])/dt/2;
                    c2[k,l] = c2[k,l] + X[j,k].*(X[j+1,l]-X[j-1,l])/dx/2;
                end
            end
        end

        y = L*c1' - obj.A*L*c2' - obj.sigmaT*L;
        y[1,:] += obj.sigmaS*L[1,:]

        L .= L .+ dt*y;

        W1,SL = py"qr"(L); N = W1'*W; SL = SL';

        # S-step
        W .= W1; X .= X1;
        
        UN,SN,VN = svd(N)
        invN = UN * Diagonal(1 ./ SN) * VN'
        UM,SM,VM = svd(M)
        invM = UM * Diagonal(1 ./ SM) * VM'
        
        alpha = 1 ./ ( 1 ./ norm(invN) + 1 ./ norm(invM));
    
        N = invN ./ norm(invN);
        M = invM' ./ norm(invM);

        S = alpha*(SK*N +M*SL);
      
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t,0.5*sqrt(obj.gamma[1])*X*S*W';

end

function SolveChristianAdaptive(obj::DLRSolver)
    t = 0.0;
    rMaxTotal = 100;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    dx = obj.x[2]-obj.x[1];
    Nx = obj.settings.NCells;
    r = obj.settings.r; # DLR rank
    N = obj.settings.nPN; # here, N is the number of quadrature points

    r = obj.settings.r;

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = diagm(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    L = zeros(N,r);

    c1 = zeros(r,r);
    c2 = zeros(r,r);
    
    Nt = Integer(round(tEnd/dt));
    alphaVec = zeros(2,Nt)

    prog = Progress(Nt,1)
    for n = 1:Nt
        ############## Streaming ##############

        ###### K-step ######
        K = X*S;

        y = zeros(Nx,r);
        WAW = W'*obj.A*W;
        for j = 2:(obj.settings.NCells-1) # leave out ghost cells
            y[j,:] = (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2- (WAW*(K[j+1,:]-K[j-1,:])/dx/2);
            # add source term and scattering
            y[j,:] += - obj.sigmaT*K[j,:];
            y[j,:] += (obj.sigmaS*K[j,:]*W[1,:]')'*W[1,:]
        end

        K .= K + dt*y;

        X1hat,SKhat = py"qr"([K X]); 
        X1 = X1hat[:,1:r]
        SK = SKhat[1:r,1:r]
        M1hat = X1'*X1hat; Mhat = X'*X1hat;

        ###### L-step ######
        L = W*S';

        c1 = zeros(r,r);
        c2 = zeros(r,r);
        
        for k = 1:r
            for l = 1:r
                for j = 2:(obj.settings.NCells-1)
                    c1[k,l] = c1[k,l] + X[j,k].*(X[j+1,l]-2*X[j,l]+X[j-1,l])/dt/2;
                    c2[k,l] = c2[k,l] + X[j,k].*(X[j+1,l]-X[j-1,l])/dx/2;
                end
            end
        end

        y = L*c1' - obj.A*L*c2' - obj.sigmaT*L;
        y[1,:] += obj.sigmaS*L[1,:]

        L .= L .+ dt*y;

        W1,SL = py"qr"(L);
        W1hat,SLhat = py"qr"([L W]); 
        W1 = W1hat[:,1:r]
        SL = SLhat[1:r,1:r]
        Nhat = W1hat'*W; N1hat = W1hat'*W1;

        # S-update        
        invM = pinv(Mhat); invM1 = pinv(M1hat)
        invN = pinv(Nhat); invN1 = pinv(N1hat)
        
        alpha = 1 ./ ( 1 ./ norm(invN1) ./ norm(invM) + 1 ./ norm(invM1) ./ norm(invN));

        S = alpha*(invM1*SK*invN ./ norm(invM1) ./ norm(invN) +invM*SL'*invN1 ./ norm(invN1) ./ norm(invM));

        alphaVec[1,n] = t;
        alphaVec[2,n] = alpha;

        ################## truncate ##################

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(S);
        rmax = -1;
        S .= zeros(size(S));

        tmp = 0.0;
        tol = obj.settings.epsAdapt*norm(D);
        
        rmax = Int(floor(size(D,1)/2));
        
        for j=1:2*rmax
            tmp = sqrt(sum(D[j:2*rmax]).^2);
            if(tmp<tol)
                rmax = j;
                break;
            end
        end
        
        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,2);

        for l = 1:rmax
            S[l,l] = D[l];
        end

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        # update solution with new rank
        XNew = X1hat*U;
        WNew = W1hat*V;

        # update solution with new rank
        S = S[1:rmax,1:rmax];
        X = XNew[:,1:rmax];
        W = WNew[:,1:rmax];

        # update rank
        r = rmax;
      
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t,0.5*sqrt(obj.gamma[1])*X*S*W',alphaVec;

end


function SolveBUG(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;
    dx = obj.x[2]-obj.x[1];
    Nx = obj.settings.NCells;
    r = obj.settings.r; # DLR rank
    N = obj.settings.nPN; # here, N is the number of quadrature points

    r = obj.settings.r;

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = diagm(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    L = zeros(N,r);
    
    Nt = Integer(round(tEnd/dt));

    e1vec = zeros(N); e1vec[1] = 1.0; E1 = Diagonal(e1vec);

    prog = Progress(Nt,1)
    for n = 1:Nt

        ############## Streaming ##############

        ###### K-step ######
        K = X*S;

        y = zeros(Nx,r);
        dt = obj.settings.dt;
        dx = obj.settings.dx;
        WAW = W'*obj.A*W;
        for j = 2:(obj.settings.NCells-1) # leave out ghost cells
            y[j,:] = (K[j+1,:]-2*K[j,:]+K[j-1,:])/dt/2- (WAW*(K[j+1,:]-K[j-1,:])/dx/2);
            # add source term and scattering
            y[j,:] += - obj.sigmaT*K[j,:];
            y[j,:] += (obj.sigmaS*K[j,:]*W[1,:]')'*W[1,:]
        end

        K .= K + dt*y;

        X1,_ = py"qr"(K); M = X1'*X;

        ###### L-step ######
        L = W*S';

        dt = obj.settings.dt;
        dx = obj.settings.dx;

        c1 = zeros(r,r);
        c2 = zeros(r,r);
        
        for k = 1:r
            for l = 1:r
                for j = 2:(obj.settings.NCells-1)
                    c1[k,l] = c1[k,l] + X[j,k].*(X[j+1,l]-2*X[j,l]+X[j-1,l])/dt/2;
                    c2[k,l] = c2[k,l] + X[j,k].*(X[j+1,l]-X[j-1,l])/dx/2;
                end
            end
        end

        y = L*c1' - obj.A*L*c2' - obj.sigmaT*L;
        y[1,:] += obj.sigmaS*L[1,:]

        L .= L .+ dt*y;

        W1,_ = py"qr"(L); N = W1'*W;

        # S-step
        W .= W1; X .= X1;

        c1 = zeros(r,r);
        c2 = zeros(r,r);
        
        for k = 1:r
            for l = 1:r
                for j = 2:(obj.settings.NCells-1)
                    c1[k,l] = c1[k,l] + X[j,k].*(X[j+1,l]-2*X[j,l]+X[j-1,l])/dt/2;
                    c2[k,l] = c2[k,l] + X[j,k].*(X[j+1,l]-X[j-1,l])/dx/2;
                end
            end
        end

        S .= M*S*N';

        S .= S .+ dt*c1*S - dt*c2*S*(W'*obj.A*W) .- dt*obj.sigmaT*S .+ dt*obj.sigmaS*S*(W'*E1*W)
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t,0.5*sqrt(obj.gamma[1])*X*S*W';

end
