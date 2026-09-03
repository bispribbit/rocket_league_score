function median(arr, n,   i, tmp, c) {
  c = 0; for (i in arr) tmp[++c] = arr[i]
  asort(tmp)
  if (c == 0) return 0
  if (c % 2) return tmp[int(c/2)+1]
  return (tmp[c/2] + tmp[c/2+1]) / 2.0
}
BEGIN { FS="," }
# pass 1: performance
FNR==NR {
  if (FNR==1) next
  key = $1 SUBSEP $2
  matched[key] = $4; goals[key] = $5
  poss[key] = ($7 > 0) ? $6/$7 : 0
  next
}
# pass 2: predictions
{
  if (FNR==1) next
  key = $1 SUBSEP $2
  if (!(key in matched) || matched[key] != 1) { unjoined++; next }
  rid = $1
  n = ++cnt[rid]
  P[rid,n] = $3; T[rid,n] = $4; PO[rid,n] = poss[key]; G[rid,n] = goals[key]
  S[rid,n] = $2
  kept++
}
END {
  for (rid in cnt) {
    m = cnt[rid]
    if (m < 2) continue
    delete pa; delete ta; delete oa
    for (i=1;i<=m;i++){ pa[i]=P[rid,i]; ta[i]=T[rid,i]; oa[i]=PO[rid,i] }
    mp = median(pa); mt = median(ta); mo = median(oa)
    # possession rank within lobby (1 = highest)
    for (i=1;i<=m;i++){
      r = 1
      for (j=1;j<=m;j++) if (PO[rid,j] > PO[rid,i]) r++
      possrank = r
      margin = P[rid,i] - mp
      lmargin = T[rid,i] - mt
      omargin = PO[rid,i] - mo
      isacct = (lmargin >= 150) ? 1 : 0
      istop = (possrank == 1) ? 1 : 0
      printf "%.4f\t%.4f\t%.6f\t%d\t%d\t%d\t%d\n", margin, lmargin, omargin, isacct, istop, possrank, G[rid,i]
      total++
      acct += isacct; top += istop; both += (isacct && istop)
    }
  }
  printf "# joined=%d unjoined=%d players=%d acct_pos=%d (%.2f%%) top_poss=%d (%.2f%%) both=%d (%.2f%%)\n",
    kept, unjoined, total, acct, 100*acct/total, top, 100*top/total, both, 100*both/total > "/dev/stderr"
}
