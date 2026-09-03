# Does "this 20 seconds you played like a Champion" track how they actually played?
#
# Input: data/segment_validity*.csv from `examples/segment_validity`.
#
# For each (replay, player) with >= 3 segments, compares every pair of that player's own
# segments: did the segment the model scored higher also have higher possession? This holds
# account rank, lobby level, playstyle and match constant -- only the 20-second window varies.
#
#   0.500 => the per-segment readout is noise around a per-player average (UI is decorative)
#   >0.500 => it tracks real within-game variation
#
# A shuffled control (predictions permuted within each player) must land at 0.500; if it does
# not, the measurement is broken rather than the model.
BEGIN { FS=","; srand(20260902) }
FNR==1 { next }
{
  k = $1 SUBSEP $2
  n = ++c[k]
  P[k,n] = $7 + 0        # prediction_mmr
  O[k,n] = $9 + 0        # possession_share
  S[k,n] = $10 + 0       # mean_speed_uu
}
END {
  for (k in c) {
    m = c[k]
    if (m < 3) continue
    players++
    # observed: concordant pairs on possession, and on speed
    for (i = 1; i <= m; i++) {
      for (j = i+1; j <= m; j++) {
        if (P[k,i] == P[k,j]) continue
        # possession
        if (O[k,i] != O[k,j]) {
          pairs_o++
          hp = (P[k,i] > P[k,j]) ? i : j
          lo = (hp == i) ? j : i
          if (O[k,hp] > O[k,lo]) conc_o++
        }
        # speed
        if (S[k,i] != S[k,j]) {
          pairs_s++
          hp = (P[k,i] > P[k,j]) ? i : j
          lo = (hp == i) ? j : i
          if (S[k,hp] > S[k,lo]) conc_s++
        }
      }
    }
    # shuffled control: permute this player's predictions
    for (i = 1; i <= m; i++) perm[i] = P[k,i]
    for (i = m; i > 1; i--) { r = int(rand()*i)+1; t = perm[i]; perm[i] = perm[r]; perm[r] = t }
    for (i = 1; i <= m; i++) {
      for (j = i+1; j <= m; j++) {
        if (perm[i] == perm[j] || O[k,i] == O[k,j]) continue
        pairs_c++
        hp = (perm[i] > perm[j]) ? i : j
        lo = (hp == i) ? j : i
        if (O[k,hp] > O[k,lo]) conc_c++
      }
    }
    # how much does the model's own output vary within this player's game?
    mn = 1e9; mx = -1e9; sum = 0
    for (i = 1; i <= m; i++) { v = P[k,i]; if (v<mn) mn=v; if (v>mx) mx=v; sum += v }
    spread_sum += (mx - mn); spread_n++
    if (mx - mn < 1.0) flat++
  }
  printf "player-games (>=3 segments) : %d\n", players
  printf "  model output is FLAT      : %d (%.1f%%)  [max-min < 1 MMR across the game]\n", flat, 100*flat/players
  printf "  mean within-game spread   : %.1f MMR\n", spread_sum/spread_n
  printf "\nwithin-player segment concordance (0.500 = readout is noise)\n"
  printf "  vs possession             : %.4f over %d pairs\n", conc_o/pairs_o, pairs_o
  printf "  vs mean speed             : %.4f over %d pairs\n", conc_s/pairs_s, pairs_s
  printf "  shuffled control          : %.4f over %d pairs   <- must be ~0.500\n", conc_c/pairs_c, pairs_c
}
