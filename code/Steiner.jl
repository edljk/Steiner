module Steiner

using CPLEX, MathProgBase, Gurobi
using Base.LinAlg, NearestNeighbors, MAT
using GeneralTools, GraphicTools, MeshTools, ColorsTools
include(string(ENV["HOME"],"/Julia/Steiner/code/Steinersetup.jl"))

export steinerLP, steinerCPLEX


#----------------------------------------------------------
function steinerPROX(np::Int64=400,steinertype::Int64=3;
  nearestk::Int64=40,dim::Int64=2,npt::Int64=4,savedata::Bool=false,maxiter::Int64=1000)

  Main.closeall()
  # for nn = 3:10 Steiner.steinerCPLEX(500,3,npt=nn,dim=2,nearestk=50,savedata=true) end
  # for nn in [4,8] Steiner.steinerCPLEX(500,3,npt=nn,dim=3,nearestk=50,savedata=true) end
  # for kk in [4,3] for nn in [4,6,8] Steiner.reloadsteiner(2000,kk,nearestk=50,npt=nn,dim=3) end end
  targetp, npt = define_targets(dim,npt)

  # Mesh
  # 1 <-> delaunay of random points in [-1,1]
  # 2 <-> Nearest neighbor type  [-1,1]
  # 3 <-> Nearest neighbor on a grid of [-1,1]
  p, e, np, ne = gensteiner_mesh(steinertype,targetp,np,nearestk=nearestk,dim=dim)
  l = sqrt(sum((p[e[:,1],:]- p[e[:,2],:]).^2, 2))

  #plot_graph(p,e,e_scalars=ones(size(e,1)),figure=1);plot_points(targetp[:,1:dim],(0.5,0.,0.),scales=0.08,figure=1);view2D()

  # divergence matrix
  A = sparse(e[:,1], collect(1:ne),  ones(ne), np, ne) +
  sparse(e[:,2], collect(1:ne), -ones(ne), np, ne)
  L = hcat(speye(ne,ne),-speye(ne,ne))
  b = zeros(np,npt);

  # Dirichlet condition
  targets = (np-size(targetp,1)+1):np
  b[targets,:] = eye(size(targetp,1))
  b[end,:] = -ones(1,npt); b[end,end] = 0 # sink
  # varables
  u = zeros(ne, npt);xi = zeros(np,npt);uout = copy(u)
  xi_old = copy(xi); u_old = copy(u)

  # preconditioning
  fact = 0.1; tau = fact/2.
  sigma = 1./sum(A.^2, 2)/fact;rho = 1.9; dykstra_iter = 1

  # main loop
  dims = [size(u)...];check = 100
  for iter = 1:maxiter
    copy!(xi_old,xi); copy!(u_old,u)

    # dual update
    for j = 1:npt
      xi[:,j] = xi[:,j] + sigma.*(A*u[:,j] - b[:,j]);
    end
    xi_ = 2*xi - xi_old

    # primal update
    for j = 1:npt
      u[:,j] = u[:,j] - tau.*(A'*xi_[:,j]);
    end
    ccall((:main_shrink,"SteinerADMM.so"),Int32,(Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Int64}),u,tau*l,uout,dims);
    copy!(u,uout)

    # overrelaxation
    xi = (1-rho)*xi_old + rho*xi
    u = (1-rho)*u_old + rho*u

    if mod(iter, check) == 0
      div = maximum(abs(A*u-b))
      t = maximum(u,2) - minimum(u,2)
      f = t'*l
      @printf "iter = %04d, length = %f, ||div||_infty = %f, dykstra_iter = %f\n"  iter f[1] div dykstra_iter
    end

  end

  # display sizes
  println("np              = "*InBlue*string(np)*InDefault)
  println("ne              = "*InBlue*string(ne)*InDefault)
  println("dims of A       = "*InBlue*string(size(A))*InDefault)
  t = maximum(u,2) - minimum(u,2); s = zeros(ne)
  displaysave_steinertree(t[:,1]+0.3,s,e,p,targetp,1,savedata,steinertype,nearestk)
end

#----------------------------------------------------------
function steinerCPLEX(np::Int64=400,steinertype::Int64=1;
  nearestk::Int64=20,dim::Int64=2,npt::Int64=4,savedata::Bool=false)

  Main.closeall();
  # for nn = 3:10 Steiner.steinerCPLEX(500,3,npt=nn,dim=2,nearestk=50,savedata=true) end
  # for nn in [4,8] Steiner.steinerCPLEX(500,3,npt=nn,dim=3,nearestk=50,savedata=true) end
  # for kk in [4,3] for nn in [4,6,8] Steiner.reloadsteiner(2000,kk,nearestk=50,npt=nn,dim=3) end end
  targetp, npt = define_targets(dim,npt)

  # Mesh
  # 1 <-> delaunay of random points in [-1,1]
  # 2 <-> Nearest neighbor type  [-1,1]
  # 3 <-> Nearest neighbor on a grid of [-1,1]
  p, e, np, ne = gensteiner_mesh(steinertype,targetp,np,nearestk=nearestk,dim=dim)
  l = sqrt(sum((p[e[:,1],:]- p[e[:,2],:]).^2, 2))
  #plot_graph(p,e,e_scalars=ones(size(e,1)),figure=1);plot_points(targetp[:,1:dim],(0.5,0.,0.),scales=0.08,figure=1);view2D()

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
  #solver = CplexSolver(CPX_PARAM_BARCROSSALG=-1,CPX_PARAM_LPMETHOD=2,CPX_PARAM_SCRIND=1,CPX_PARAM_SIMDISPLAY=1)
  solver = CplexSolver(CPX_PARAM_BARCROSSALG=1,CPX_PARAM_LPMETHOD=4,CPX_PARAM_SCRIND=1,CPX_PARAM_SIMDISPLAY=1)
  #solver = GurobiSolver(DisplayInterval=10,Crossover=4,Method=1)
  LB = -Inf*ones(length(cc)); UB =  Inf*ones(length(cc))
  I0 = sub2ind((ne,npt+2),1:ne,repmat([npt,],ne))
  LB[I0] = 0.;  UB[I0] = 0.
  sol = linprog(cc[1,:],vcat(A,Amaxmin),'<',vcat(b,zeros(size(Amaxmin,1)))[:,1],LB,UB, solver)
  objval = sol.objval
  u = sol.sol

  # display
  feas = sqrt(sum((A*u-b).^2,2))[1]
  @printf "feas            = %4.6f\n"  feas
  println("objv            = "*InOrange*"$(objval)"*InDefault)

  t = u[(ne*npt+1):(ne*(npt+1))]
  s = u[(ne*(npt+1)+1):end]
  displaysave_steinertree(t,s,e,p,targetp,1,savedata,steinertype,nearestk)

  # display sizes
  println("np              = "*InBlue*string(np)*InDefault)
  println("ne              = "*InBlue*string(ne)*InDefault)
  println("dims of A       = "*InBlue*string(size(A))*InDefault)
  println("dims of Amaxmin = "*InBlue*string(size(Amaxmin))*InDefault)
  println("dims of c       = "*InBlue*string(length(cc))*InDefault)
  println("dims of btot    = "*InBlue*string(length(vcat(b,zeros(size(Amaxmin,1)))))*InDefault)
  println("max last coord  = "*InBlue*string(maximum(abs(u[I0])))*InDefault)

end

#----------------------------------------------------------
function steinerLP(maxiter::Int64=10000,np::Int64=400)

  # Target points
  Main.closeall(); npt = 4;dim = 2
  tt = linspace(0.,2*pi,npt+1)[1:(end-1)]
  targetp = hcat(cos(tt),sin(tt),zeros(npt))*.95

  # Mesh
  steinertype = 1 # 1 <-> delaunay of random points in [-1,1]
  # 2 <-> Nearest neighbor type  [-1,1]
  # 3 <-> Nearest neighbor on a grid of [-1,1]
  p, e, np, ne = gensteiner_mesh(steinertype,targetp,np,nearestk=8)
  l = sqrt(sum((p[e[:,1],:]- p[e[:,2],:]).^2, 2))
  #plot_graph(p,e,e_scalars=ones(size(e,1)),figure=1);plot_points(targetp[:,1:dim],(0.5,0.,0.),scales=0.08,figure=1);view2D()

  # divergence matrix
  A = sparse(e[:,1], collect(1:ne),  ones(ne), np, ne) +
  sparse(e[:,2], collect(1:ne), -ones(ne), np, ne)
  L = hcat(speye(ne,ne),-speye(ne,ne))
  b = zeros(np,npt); Ap = A'; Lp = L'

  # Dirichlet condition
  targets = (np-size(targetp,1)+1):np
  b[targets,:] = eye(size(targetp,1))
  b[end,:] = -ones(1,npt); b[end,end] = 0 # sink

  # parameters
  Lip = 5; tau = 0.1/Lip; sigma = 1/tau/Lip^2

  # memory allocation
  u = zeros(ne, npt);s = zeros(ne,1); t = zeros(ne,1)     # primals
  psi = zeros(ne,npt); phi = zeros(ne,npt);xi = zeros(np,npt) # duals
  psi_ = copy(psi); phi_ = copy(phi); xi_ = copy(xi)
  prodAtmp = zeros(size(A,1)); prodAptmp = zeros(size(A,2))
  prodLtmp_T = zeros(size(L,1));prodLtmp_S = zeros(size(L,1))
  prodLptmp_T = zeros(size(L,2));prodLptmp_S = zeros(size(L,2))
  uS = copy(vcat(u[:,1],s)); uT = copy(vcat(u[:,1],t))

  for iter = 1:maxiter
    copy!(psi_,psi); copy!(phi_,phi); copy!(xi_,xi)
    #psi_ = copy(psi); phi_ = copy(phi); xi_ = copy(xi)
    for j = 1:npt
      psi[:,j] = max(0, psi[:,j] + sigma*(L*vcat(u[:,j],s)))
      phi[:,j] = min(0, phi[:,j] + sigma*(L*vcat(u[:,j],t)))
      xi[:,j] = xi[:,j] + sigma*(A*u[:,j] - b[:,j])
    end
    psi_ = copy(2*psi - psi_); phi_ = copy(2*phi - phi_); xi_ = copy(2*xi - xi_)
    for j = 1:npt
      tmp_s = L'*psi_[:,j]
      tmp_t = L'*phi_[:,j]

      u[:,j] = u[:,j] - tau*(tmp_s[1:ne] + tmp_t[1:ne] + A'*xi_[:,j])
      s = s - tau*(l + tmp_s[ne+1:end])
      t = t - tau*(-l + tmp_t[ne+1:end])
    end

    # display
    if mod(iter,1000)==0
      feas = sqrt(sum((A*u-b).^2,2))[1]
      @printf "iter = %04d, feas = %4.6f\n" iter feas
      #displaysave_steinertree(s[:,1],t[:,1],e,p,targetp,1)
      ve = 1 - max(0,s-t)
      minmaxmean(ve)
      Ie = find(x->x>0.1,ve)
      mlab.figure(2);mlab.clf();vec = plot_graph(p,e[Ie,:],e_scalars=1-ve[Ie,1],figure=2);view2D();setcolormap(vec,"Oranges")
      plot_points(targetp[:,1:dim],(0.5,0.,0.),scales=0.08,figure=2)
    end
  end
end

#----------------------------------------------------------
function displaysave_steinertree(t::Array{Float64,1},s::Array{Float64,1},e::Array{Int64,2},
  p::Array{Float64,2},targetp::Array{Float64,2},fignum::Int64,savedata::Bool,
  steinertype::Int64,nearestk::Int64)
  minmaxmean(t-s)
  dim = size(p,2); np = size(p,1); npt = size(targetp,1); ve = 1 - max(0,t-s)
  mlab.figure(fignum);mlab.clf();
  opacity1 = 0.8
  if dim==3
    opacity1 = 0.01; opacity2 = 1.; scalev = 0.12
    #vec1 = plot_graph(p,e,e_scalars=ve,figure=fignum,line_width=1.2,color=(0.9,0.3,0.),opacity=opacity1)
  else
    opacity1 = 0.05; opacity2 = 1.; scalev = 0.08
    vec1 = plot_graphglyph(p,e,ve,figure=fignum,line_width=0.8,color=(0.9,0.3,0.),opacity=opacity1)
  end
  #esm = Int64(size(e,1)/2); ep = e[1:esm,:]
  #ve = min(ve[1:esm],ve[(esm+1):end])
  ep = e
  Ie = find(x->x<0.9,ve)
  println(length(Ie))
  if dim==3
    vec2 = plot_graph(p,ep[Ie,:],e_scalars=1-ve[Ie,1],figure=fignum,tube_radius=0.03,vmin=0.,vmax=1.,opacity=opacity2)
  else
    vec2 = plot_graph(p,ep[Ie,:],e_scalars=1-ve[Ie,1],figure=fignum,tube_radius=0.015,vmin=0.,vmax=1.,opacity=opacity2)
    #vec2 = plot_graphglyph(p,ep[Ie,:],ve[Ie,1],line_width=2.8,figure=fignum,vmin=0.,vmax=1.,opacity=opacity2)
  end
  println(vec2);colb = mycolorbar(vec2,nblabels=8); setcolormap(vec2,"Oranges",reversecolors=true)
  #colb["reverse_lut"] = true
  if dim==2 view2D() end
  if dim==3 plot_points(p[unique(ep[Ie,:][:]),:],(0.9,0.9,0.9),scales=0.08,figure=fignum) end
  plot_points(targetp[:,1:dim],(0.5,0.,0.),scales=scalev,figure=fignum);
  plot_points([0. 0. 0.],(0.1,0.9,0.),scales=0.05,figure=fignum)
  #vec2["actor"]["property"]["interpolation"] = "phong"
  #vec2["actor"]["property"]["specular"] = 0.9
  #vec2["actor"]["property"]["specular_power"] = 50

  ff = mlab.gcf()
  ff["scene"]["render_window"]["aa_frames"] = 8
  # save
  if savedata
    if dim==3 lightvtk() end
    filename = steinerfilename(dim,np,npt,steinertype,nearestk)
    println(InMagenta*filename*InDefault)
    if dim==2 mlab.savefig(filename*".png",size=(2000,2000)) end
    if dim==3
      #savefigs("/tmp/im_",80,figure=1,size=(500,500),sleeptime=0.,
      #genmovie=true,filemovie=filename,delaymovie=10.)
    end
    matwrite(replace(filename,"pictures","runs")*".mat",Dict("t"=>t,"s"=>s,"p"=>p,"e"=>e,"targetp"=>targetp))
  end
end

#----------------------------------------------------------
function reloadsteiner(np::Int64=400,steinertype::Int64=3;
  nearestk::Int64=20,dim::Int64=2,npt::Int64=4,
  savedata::Bool=false)
  targetp, npt = define_targets(dim,npt)
  p, e, np, ne = gensteiner_mesh(steinertype,targetp,np,nearestk=nearestk,dim=dim)
  filename = steinerfilename(dim,np,npt,steinertype,nearestk)
  println(InMagenta*filename*InDefault)
  file = matopen(string(replace(filename,"pictures","runs"),".mat"))
  p = read(file,"p"); t = read(file, "t"); s = read(file,"s"); e = read(file,"e"); targetp = read(file,"targetp")
  close(file)
  l = sqrt(sum((p[e[:,1],:]- p[e[:,2],:]).^2, 2)); ve = 1 - max(0,t-s)
  println("objv            = "*InOrange*string(sum((t-s).*l))*InDefault)
  println("set of norms    = "*InOrange*string(sort(unique(round(ve*1e3)/1000),rev=true))*InDefault)
  minmaxmean(ve)
  displaysave_steinertree(t,s,e,p,targetp,1,true,steinertype,nearestk)
end

end # module
