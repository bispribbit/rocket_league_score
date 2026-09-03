function median(arr,   i, tmp, c) {
  c=0; for (i in arr) tmp[++c]=arr[i]; asort(tmp)
  if (c==0) return 0
  if (c%2) return tmp[int(c/2)+1]
  return (tmp[c/2]+tmp[c/2+1])/2.0
}
BEGIN { FS="," }
FNR==NR { if(FNR==1) next; k=$1 SUBSEP $2; mt[k]=$4; go[k]=$5; po[k]=($7>0)?$6/$7:0; next }
{ if(FNR==1) next; k=$1 SUBSEP $2
  if(!(k in mt) || mt[k]!=1) next
  r=$1; n=++c[r]; P[r,n]=$3; T[r,n]=$4; PO[r,n]=po[k]; G[r,n]=go[k] }
END {
  for (r in c) {
    m=c[r]; if (m<3) continue
    delete pa; delete ta
    for(i=1;i<=m;i++){pa[i]=P[r,i]; ta[i]=T[r,i]}
    mp=median(pa); mt2=median(ta)
    # is this a mixed lobby (someone >=150 above median label)?
    mixed=0; for(i=1;i<=m;i++) if (T[r,i]-mt2>=150) mixed=1
    # our pick = highest margin
    best=1; for(i=2;i<=m;i++) if (P[r,i]>P[r,best]) best=i
    # is pick top possession?
    ptop=1; for(i=1;i<=m;i++) if (PO[r,i]>PO[r,ptop]) ptop=i
    # is pick top goals (unique)?
    gtop=1; tie=0; for(i=1;i<=m;i++){ if(G[r,i]>G[r,gtop]){gtop=i;tie=0} else if(i!=gtop && G[r,i]==G[r,gtop]) tie=1 }
    lob++
    pick_dom += (best==ptop)?1:0
    pick_acct += (T[r,best]-mt2>=150)?1:0
    pick_both += ((best==ptop) && (T[r,best]-mt2>=150))?1:0
    pick_gtop += ((best==gtop)&&!tie)?1:0
    chance += 1.0/m
    if (mixed) { mlob++
      mpick_dom += (best==ptop)?1:0
      mpick_acct += (T[r,best]-mt2>=150)?1:0
      mpick_both += ((best==ptop)&&(T[r,best]-mt2>=150))?1:0
      mchance += 1.0/m }
  }
  printf "ALL LOBBIES (n=%d)\n", lob
  printf "  pick is top-possession    : %5.1f%%   (chance %.1f%%)\n", 100*pick_dom/lob, 100*chance/lob
  printf "  pick is unique top scorer : %5.1f%%   (chance %.1f%%)\n", 100*pick_gtop/lob, 100*chance/lob
  printf "  pick is account-positive  : %5.1f%%\n", 100*pick_acct/lob
  printf "  pick is BOTH              : %5.1f%%\n", 100*pick_both/lob
  printf "\nMIXED LOBBIES ONLY (n=%d)\n", mlob
  printf "  pick is top-possession    : %5.1f%%   (chance %.1f%%)\n", 100*mpick_dom/mlob, 100*mchance/mlob
  printf "  pick is account-positive  : %5.1f%%\n", 100*mpick_acct/mlob
  printf "  pick is BOTH              : %5.1f%%\n", 100*mpick_both/mlob
}
