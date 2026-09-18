def Theta(Knot):
    [Cs,phi] = Knot
    R.<t1,t2> = PolynomialRing(QQ); Fr=Frac(R);
    n=Cs.nrows()
    A1 = identity_matrix(Fr,2*n+1)
    for c in Cs: A1[c[1],c[1]+1]=-t1^c[0];A1[c[1],c[2]+1]=t1^c[0]-1;A1[c[2],c[2]+1]=-1
    Delta1 = t1^((-sum(phi)-sum(Cs[:,0])[0])/2)*det(A1);Delta2=Delta1(t1=t2);Delta3 = Delta1(t1=t1*t2)
    G1=A1.inverse()
    G2 = G1(t1=t2);G3=G1(t1=t1*t2);
    def F1(s,i,j): return s*(
    1/2+t2^s*G1[i,i]*G2[j,i]+((t1^s-1)*t2^(2*s)*G1[j,i]*G2[j,i])/(t2^s-1)-G1[i,i]*G2[j,j]-(t1^s-1)*t2^s*G1[j,i]*G2[j,j]/(t2^s-1)-G3[i,i]-(t2^s-1)*G2[j,i]*G3[i,i]+2*G2[j,j]*G3[i,i]+(t1^s*t2^s-1)*G3[j,i]/(t2^s-1)-t2^s*(t1^s*t2^s-1)*G1[i,i]*G3[j,i]/(t2^s-1)-(t1^s-1)*(t2^s+1)*(t1^s*t2^s-1)*G1[j,i]*G3[j,i]/(t2^s-1)+(t1^s*t2^s-1)*G2[i,j]*G3[j,i]/(t2^s-1)+(t1^s*t2^s-1)*G2[j,i]*G3[j,i]+(t2^s-2)*(t1^s*t2^s-1)*G2[j,j]*G3[j,i]/(t2^s-1)+G1[i,i]*G3[j,j]+(t1^s-1)*t2^s*G1[j,i]*G3[j,j]/(t2^s-1)-G2[i,i]*G3[j,j]-t2^s*G2[j,i]*G3[j,j]
    )
    def F2(s0,i0,j0,s1,i1,j1): return s1*( (t1^s0-1)*t2^s0*(t1^s1*t2^s1-1)*G1[j1,i0]*G2[i1,i0]*G3[j0,i1] - (t1^s0-1)*(t1^s1*t2^s1-1)*G1[j1,i0]*G2[i1,j0]*G3[j0,i1] -(t1^s0-1)*t2^s0*(t1^s1*t2^s1-1)*G1[j1,i0]*G2[j1,i0]*G3[j0,i1]+(t1^s0-1)*(t1^s1*t2^s1-1)*G1[j1,i0]*G2[j1,j0]*G3[j0,i1]   )/(t2^s1-1)
    def Gam1(ph,k): return ph*G3[k,k]-ph/2
    theta = 0; 
    for c in Cs: theta+= F1(c[0],c[1],c[2])
    for c1 in Cs: 
        for c2 in Cs: theta+= F2(c1[0],c1[1],c1[2],c2[0],c2[1],c2[2])
    for k in range(2*n+1): theta+=Gam1(phi[k],k)
    theta=theta*Delta1*Delta2*Delta3
    return theta
