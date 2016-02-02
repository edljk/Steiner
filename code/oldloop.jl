




        #=copy!(psi_,psi); copy!(phi_,phi); copy!(xi_,xi)
        #psic = copy(psi); phi = copy(phi); xic = copy(xi)
        for j = 1:q
            copy!(uS,1,u[:,j],1,size(u,1)); copy!(uS,size(u,1)+1,s,1,length(s))
            copy!(uT,1,u[:,j],1,size(u,1)); copy!(uT,size(u,1)+1,t,1,length(t))
            A_mul_B!(prodLtmp_S,L,uS); A_mul_B!(prodLtmp_T,L,uT)
            for k = 1:size(psi,1)
                psi[k,j] = max(0., psi[k,j] + sigma*prodLtmp_S[k])
                phi[k,j] = min(0., phi[k,j] + sigma*prodLtmp_T[k])
            end
            A_mul_B!(prodAtmp,A,u[:,j])
            for k = 1:size(xi,1)
                xi[k,j] += sigma*(prodAtmp[k] - b[k,j])
            end
        end
        #for j = 1:q        
        #    psic[:,j] = max(0, psic[:,j] + sigma*(L*vcat(u[:,j],s)));
        #    phic[:,j] = min(0, phic[:,j] + sigma*(L*vcat(u[:,j],t)));
        #    xic[:,j] = xic[:,j] + sigma*(A*u[:,j] - b[:,j]);
        #end
        #println(maximum(abs(psi-psic)),"---",maximum(abs(phi-phic)),"---",maximum(abs(xi-xic)))
        

        for j = 1:size(psi,2)
            for i = 1:size(psi,1)
                psi_[i,j] = 2*psi[i,j] - psi_[i,j]
                phi_[i,j] = 2*phi[i,j] - phi_[i,j]
            end
        end
        for k = 1:length(xi)
            xi_[k] = 2*xi[k] - xi_[k]
        end

        #tc = copy(t); sc = copy(s)
        for j = 1:q 
            A_mul_B!(prodLptmp_S,Lp,psi_[:,j])
            A_mul_B!(prodLptmp_T,Lp,phi_[:,j])
            A_mul_B!(prodAptmp,Ap,xi_[:,j])
            for k = 1:size(u,1)
                u[k,j] = u[k,j] - tau*(prodLptmp_S[k] + prodLptmp_T[k] + prodAptmp[k])
            end
            for k = 1:length(s)
                s[k] = s[k] - tau*(l[k] + prodLptmp_S[ne+k])
                t[k] = t[k] - tau*(-l[k] + prodLptmp_T[ne+k])
            end
        end
        #for j=1:q    
        #    tmp_s = L'*psi_[:,j];
        #    tmp_t = L'*phi_[:,j];            
        #    u[:,j] = u[:,j] - tau*(tmp_s[1:ne] + tmp_t[1:ne] + A'*xi_[:,j]);
        #    sc = sc - tau*(l + tmp_s[ne+1:end]);
        #    tc = tc - tau*(-l + tmp_t[ne+1:end]);
        #end
        #println(maximum(abs(s-sc)),"---",maximum(abs(t-tc)),"---")
        
    end
