function gensteiner_mesh(steinertype::Int64=1,targetp::Array{Float64,2}=zeros(1,3),np::Int64=400;
                         dim::Int64=2,nearestk::Int64=30)
    
    if steinertype==1 # 1 <-> delaunay of random points in [-1,1]
        p = vcat(2.*rand(np-size(targetp,1),dim)-1,targetp[:,1:dim])
        tri = delaunay(p)
        ep = convert(Array{Int64,2},vcat(tri[:,[1,2]],tri[:,[2,3]],tri[:,[1,3]]))
    elseif steinertype==2 # 2 <-> Nearest neighbor type  [-1,1]
        p = vcat(2*rand(np-size(targetp,1),dim)-1,targetp[:,1:dim])
        tree = KDTree(p[:,1:dim]',Euclidean();leafsize=2)
        println(tree)
        dists = zeros(size(p,1),nearestk)
        idxs = zeros(Int64,size(p,1),nearestk)
        ep = zeros(Int64,size(p,1)*nearestk,2)
        for k = 1:size(p,1)
            idxst, distst = knn(tree,p[k,1:dim],nearestk)
            ep[((k-1)*nearestk+1):k*nearestk,1] = k
            ep[((k-1)*nearestk+1):k*nearestk,2] = idxst
            #println(idxst,"\n",distst)
        end
    elseif steinertype==3 # 3 <-> Nearest neighbor on a grid of [-1,1]
        npg = Int64(max(ceil(np^(1./dim)),5)); xx = linspace(-1,1,npg)
        if dim==2
            X, Y = Main.ndgrid(xx,xx)
            Z =zeros(size(X))
        else
            X, Y, Z = Main.ndgrid(xx,xx,xx)
        end
        pts = hcat(X[:],Y[:],Z[:]); pts = pts + (2.*rand(size(pts))-1.)/10000.
        p = vcat(pts[:,1:dim],targetp[:,1:dim])
        tree = KDTree(p[:,1:dim]',Euclidean();leafsize=2)
        println(tree)
        dists = zeros(size(p,1),nearestk)
        idxs = zeros(Int64,size(p,1),nearestk)
        ep = zeros(Int64,size(p,1)*nearestk,2)
        for k = 1:size(p,1)
            idxst, distst = knn(tree,p[k,1:dim],nearestk)
            ep[((k-1)*nearestk+1):k*nearestk,1] = k
            ep[((k-1)*nearestk+1):k*nearestk,2] = idxst
            #println(idxst,"\n",distst)
        end
    end
    ep = uniquerows(sort(ep,2))[1]
    em = hcat(ep[:,2],ep[:,1]); e = vcat(ep,em); 
    np = size(p,1);  ne = size(e,1)
    return p, e, np, ne
end

#----------------------------------------------------------
function define_targets(dim::Int64=2,npt::Int64=4)
    # Target points
    if dim==2
        tt = linspace(0.,2*pi,npt+1)[1:(end-1)]
        targetp = hcat(cos(tt),sin(tt),zeros(npt))*.95
        if npt>8
            targetp = 2*rand(npt,2) - 1.
        end
    elseif dim==3
        ST, FT, SH, FH, SO, FO, SD, FD, SI, FI = polyedres_def()
        if npt==4
            targetp = ST
        elseif npt==6
            targetp = SO     
        elseif npt==8
            targetp = SH
        elseif npt==12
            targetp = SI   
        elseif npt==20
            targetp = SD   
        end   
    end
    targetp = targetp - repmat(mean(targetp,1),size(targetp,1),1)
    targetp = 0.9*targetp/maximum(abs(targetp))
    return targetp, size(targetp,1)
end