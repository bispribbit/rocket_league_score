# cols: 1 margin 2 lmargin 3 omargin 4 isacct 5 istop 6 possrank 7 goals
BEGIN { FS="\t" }
{ m[NR]=$1; lab_acct[NR]=$4; lab_top[NR]=$5; lab_both[NR]=($4&&$5)?1:0; n=NR }
END {
  # sort indices by margin descending
  for (i=1;i<=n;i++) idx[i]=i
  # simple insertion via gawk asorti on a value-keyed map
  for (i=1;i<=n;i++) key[i] = sprintf("%016.6f_%06d", 1000000 - m[i], i)
  c=0; for (i=1;i<=n;i++) sk[++c]=key[i]
  asort(sk)
  split("acct top both", names, " ")
  for (t=1;t<=3;t++) {
    name=names[t]; tp=0; sump=0; pos=0
    for (i=1;i<=n;i++) { split(sk[i],parts,"_"); j=parts[2]+0
      lab = (name=="acct")?lab_acct[j]:((name=="top")?lab_top[j]:lab_both[j])
      if (lab) { tp++; sump += tp/i; pos++ }
    }
    if (pos>0) printf "%-6s positives=%-5d base_rate=%.4f  AP=%.4f  lift=%.2fx\n", name, pos, pos/n, sump/pos, (sump/pos)/(pos/n)
  }
}
