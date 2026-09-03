function median(arr,   i,tmp,c){c=0;for(i in arr)tmp[++c]=arr[i];asort(tmp);if(c==0)return 0;if(c%2)return tmp[int(c/2)+1];return (tmp[c/2]+tmp[c/2+1])/2.0}
BEGIN{FS=","}
FNR==NR{if(FNR==1)next;k=$1 SUBSEP $2;mt[k]=$4;po[k]=($7>0)?$6/$7:0;next}
{if(FNR==1)next;k=$1 SUBSEP $2;if(!(k in mt)||mt[k]!=1)next;r=$1;n=++c[r];P[r,n]=$3;T[r,n]=$4;PO[r,n]=po[k]}
END{
 for(r in c){m=c[r];if(m<3)continue
  delete ta;for(i=1;i<=m;i++)ta[i]=T[r,i]
  mt2=median(ta);np=0;for(i=1;i<=m;i++)if(T[r,i]-mt2>=150)np++
  if(np>0){ml++; sum_np+=np; sum_m+=m; chance_acct += np/m}
 }
 printf "mixed lobbies=%d  avg players=%.2f  avg account-positives per lobby=%.2f\n", ml, sum_m/ml, sum_np/ml
 printf "=> random pick would be account-positive %.1f%% of the time\n", 100*chance_acct/ml
}
