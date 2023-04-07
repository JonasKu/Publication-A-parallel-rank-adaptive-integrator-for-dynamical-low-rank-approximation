__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    Δx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    Δt::Float64;
    # CFL number 
    cfl::Float64;
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid

    # physical parameters
    σₐ::Float64;
    σₛ::Float64;

    # low rank parameters
    r::Int;

    # rank adaptivity
    ϑ::Float64;
    rMax::Int;
    rMin::Int;

    ϑIndex::Float64;
    cη::Float64;

    function Settings(Nx::Int=2002,problem::String="LineSource")
        # spatial grid setting
        NCells = Nx - 1;
        a = -5; # left boundary
        b = 5; # right boundary
        
        # time settings
        tEnd = 1.0;
        cfl = 0.99; # CFL condition
        
        # number PN moments
        nPN = 2001;

        x = collect(range(a,stop = b,length = NCells));
        Δx = x[2]-x[1];
        x = [x[1]-Δx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x .+ Δx/2;
        xMid = x[1:(end-1)].+0.5*Δx

        Δt = cfl*Δx;

        # physical parameters
        σₛ = 1.0;
        σₐ = 0.0;   

        r = 50;

        # parameters rank adaptivity
        ϑ = 1e-2;
        rMax = Int(floor(min(nPN / 2,(Nx-1) / 2)));
        rMin = 5;
        cη = 1;
        ϑIndex = 1;

        # build class
        new(Nx,NCells,a,b,Δx,tEnd,Δt,cfl,nPN,x,xMid,σₐ,σₛ,r,ϑ,rMax,rMin,ϑIndex,cη);
    end

end

function IC(obj::Settings,x,xi=0.0)
    y = zeros(size(x));
    x0 = 0.0
    s1 = 0.03
    s2 = s1^2
    floor = 1e-4
    x0 = 0.0
    for j = 1:length(y);
        y[j] = max(floor,1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
    end
    return y;
end