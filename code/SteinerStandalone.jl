module SteinerStandalone

using Base.LinAlg, NearestNeighbors, MAT
using CPLEX, MathProgBase, Gurobi
using GeneralTools, ColorsTools
include(string(ENV["HOME"],"/Julia/Tools/Mesh/polyedres.jl"))
include(string(ENV["HOME"],"/Julia/Steiner/code/Steinersetup.jl"))

#----------------------------------------------------------
function steinerCPLEX(np::Int64=400,steinertype::Int64=3;
                      nearestk::Int64=20,dim::Int64=2,npt::Int64=4,savedata::Bool=true)

    # for nn = 3:10 SteinerStandalone.steinerCPLEX(500,3,npt=nn,dim=2,nearestk=50,savedata=true) end
    # for nn in [4,6,8,12,20] SteinerStandalone.steinerCPLEX(2000,3,npt=nn,dim=3,nearestk=50,savedata=true) end
    targetp, npt = define_targets(dim,npt)

    # Mesh
    # 1 <-> delaunay of random points in [-1,1]
    # 2 <-> Nearest neighbor type  [-1,1]
    # 3 <-> Nearest neighbor on a grid of [-1,1]
    p, e, np, ne = gensteiner_mesh(steinertype,targetp,np,nearestk=nearestk,dim=dim)
    l = sqrt(sum((p[e[:,1],:]- p[e[:,2],:]).^2, 2))

    A = []; Amaxmin = []
    for k = 1:npt
        IG = sub2ind((ne,npt+2),1:ne,repmat([k,],ne))
        if k==1
            A = sparse(e[:,1], IG,  ones(ne), np, ne*(npt+2)) + 
                sparse(e[:,2], IG, -ones(ne), np, ne*(npt+2))
        else
            A = vcat(A,sparse(e[:,1], IG,  ones(ne), np, ne*(npt+2)) + 
                       sparse(e[:,2], IG, -ones(ne), np, ne*(npt+2)))
        end
    end
    # add inequalities for max/min corresponding to variables t/s
    IGt = sub2ind((ne,npt+2),1:ne,repmat([npt+1,],ne))
    IGs = sub2ind((ne,npt+2),1:ne,repmat([npt+2,],ne))
    for k = 1:npt
        IG = sub2ind((ne,npt+2),1:ne,repmat([k,],ne))        
        if k==1
            Amaxmin = sparse(1:ne,IG,ones(ne),ne,ne*(npt+2)) + sparse(1:ne,IGt,-ones(ne),ne,ne*(npt+2))
        else
            Amaxmin = vcat(Amaxmin,sparse(1:ne,IG,ones(ne),ne,ne*(npt+2)) + sparse(1:ne,IGt,-ones(ne),ne,ne*(npt+2)))
        end
        Amaxmin = vcat(Amaxmin,sparse(1:ne,IG,-ones(ne),ne,ne*(npt+2)) +  sparse(1:ne,IGs,ones(ne),ne,ne*(npt+2)))
    end      
    # Dirichlet condition
    targets = (np-size(targetp,1)+1):np
    b = zeros(np,npt); b[targets,:] = eye(size(targetp,1))
    b[end,:] = -ones(1,npt); b[end,end] = 0 # sink
    b = b[:]

    # vcost
    cc = hcat(zeros(1,npt*ne),l[:,1]',-l[:,1]')
   
    # call CPLEX/Gurobi
    solver = CplexSolver(CPX_PARAM_LPMETHOD=6,CPX_PARAM_SCRIND=1,CPX_PARAM_SIMDISPLAY=1)
    #solver = GurobiSolver(DisplayInterval=10,Method=3,Crossover=-1)
    sol = linprog(cc[1,:],vcat(A,Amaxmin),'<',vcat(b,zeros(size(Amaxmin,1)))[:,1],-Inf, Inf, solver)
    objval = sol.objval
    u = sol.sol
    
    # display
    feas = sqrt(sum((A*u-b).^2,2))[1]
    @printf "feas            = %4.6f\n"  feas
    println("objv            = "*InOrange*"$(objval)"*InDefault)
    t = u[(ne*npt+1):(ne*(npt+1))]
    s = u[(ne*(npt+1)+1):end]
    save_steinertree(t,s,e,p,targetp,1,savedata,steinertype,nearestk)
   
    
    # display sizes
    println("np              = "*InBlue*string(np)*InDefault)
    println("ne              = "*InBlue*string(ne)*InDefault)
    println("dims of A       = "*InBlue*string(size(A))*InDefault)
    println("dims of Amaxmin = "*InBlue*string(size(Amaxmin))*InDefault)
    println("dims of c       = "*InBlue*string(length(cc))*InDefault)
    println("dims of btot    = "*InBlue*string(length(vcat(b,zeros(size(Amaxmin,1)))))*InDefault)

end


#----------------------------------------------------------
function save_steinertree(t::Array{Float64,1},s::Array{Float64,1},e::Array{Int64,2},
                          p::Array{Float64,2},targetp::Array{Float64,2},fignum::Int64,savedata::Bool,
                          steinertype::Int64,nearestk::Int64)
    dim = size(p,2); np = size(p,1); npt = size(targetp,1); ve = 1 - max(0,t-s)
    #vec1 = plot_graph(p,e,ve,figure=fignum,line_width=1.2,color=(0.9,0.3,0.))
    esm = Int64(size(e,1)/2); ep = e[1:esm,:]
    ve = min(ve[1:esm],ve[(esm+1):end]); Ie = find(x->x<0.1,ve)
    minmaxmean(ve)
    println(InOrange*string(sort(unique(round(ve*1e3)/1000),rev=true))*InDefault)
    # save
    if savedata
        filename = steinerfilename(dim,np,npt,steinertype,nearestk)
        println(InMagenta*filename*InDefault)        
        matwrite(replace(filename,"pictures","runs")*".mat",Dict("t"=>t,"s"=>s,"p"=>p,"e"=>e,"targetp"=>targetp))
    end
end

end # module







