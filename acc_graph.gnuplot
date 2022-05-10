set terminal eps       
set output  "acc_graph.eps"   

set xlabel "Latency (seconds)"
set ylabel "Accuracy (%)"

set mytics
set xtics
set mxtics
set grid xtics mxtics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 2
set grid ytics mytics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 2
# set key at 700, 600
# set key spacing 1.25


# Define the broken-axis mapping
axis_gap = 500
f(x) = (x <= 4000) ? x : (x < 13000) ? NaN : (x - 9000 + axis_gap)
g(x) = (x <= 4000) ? x : (x < 4000 + axis_gap) ? NaN : (x + 9000 - axis_gap)
set xrange [0:13400] noextend
set nonlinear x via f(x) inverse g(x)
unset key

# Creation of the broken axis marks (this should be automated)
set arrow from 4000, graph 0 to 13000, graph 0 nohead lt 4000 lw 2 lc bgnd front
set arrow from 4000, graph 0 length graph  .01 angle 75 nohead lw 3 front
set arrow from 4000, graph 0 length graph -.01 angle 75 nohead lw 3 front
set arrow from 13000, graph 0 length graph  .01 angle 75 nohead lw 3 front
set arrow from 13000, graph 0 length graph -.01 angle 75 nohead lw 3 front
set arrow from screen 0.81, graph 0 length graph  .01 angle 75 nohead lw 3 front
set arrow from screen 0.81, graph 0 length graph -.01 angle 75 nohead lw 3 front
set arrow from screen 0.885, graph 0 length graph  .01 angle 75 nohead lw 3 front
set arrow from screen 0.885, graph 0 length graph -.01 angle 75 nohead lw 3 front

set arrow from 4000, graph 1 to 13000, graph 1 nohead lt 4000 lw 2 lc rgb "white" front
set arrow from 4000, graph 1 length graph  .01 angle 75 nohead lw 3 front
set arrow from 4000, graph 1 length graph -.01 angle 75 nohead lw 3 front
set arrow from 13000, graph 1 length graph  .01 angle 75 nohead lw 3 front
set arrow from 13000, graph 1 length graph -.01 angle 75 nohead lw 3 front
set arrow from screen 0.81, graph 1 length graph  .01 angle 75 nohead lw 3 front
set arrow from screen 0.81, graph 1 length graph -.01 angle 75 nohead lw 3 front
set arrow from screen 0.885, graph 1 length graph  .01 angle 75 nohead lw 3 front
set arrow from screen 0.885, graph 1 length graph -.01 angle 75 nohead lw 3 front


# set style line 1 lc rgb '#48494B' lt 1 lw 3 pt 7 ps 0.5 # gray
# set style line 2 lc rgb '#828282' lt 1 lw 3 pt 7 ps 0.5  # green
# set style line 3 lc rgb '#97978F' lt 1 lw 3 pt 7 ps 0.5 # blue
set style line 1 lc rgb '#0A1172' lt 1 lw 3 pt 7 ps 0.5 # gray
set style line 2 lc rgb '#0147AB' lt 1 lw 3 pt 7 ps 0.5  # green
set style line 3 lc rgb '#588BAE' lt 1 lw 3 pt 7 ps 0.5 # blue
set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # blue
set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # orange

set label "{/:Bold Ours}" at 750, 94.5 tc rgb '#ed7014'
set label "Falcon (Lou et al., 2020)" at 340, 78.5
set label "EVA (Dathathri et al., 2020)" at 3000, 82.5    
set label "Lora (Brutzkus et al., 2019)" at 800, 77    
set label "(Lee et al., 2021b)" at 3300, 93.5   
set arrow from 340,78 to 150, 76.9 lw 2

set arrow from 2271, 91.31 to 3703, 92.4 nohead lt 1 lw 3 lc rgb '#0A1172'
set arrow from 3703, 92.4 to 13282, 92.9 nohead dashtype 2 lt 1 lw 3 lc rgb '#0A1172'

# plot "acc.dat" using 2:3 with points ls 1 notitle
plot 'acc.dat' using 1:2 with points ls 1 notitle, \
    'acc_ours.dat' using 1:2 with linespoints ls 4 notitle, \
    'acc_lee.dat' using 1:2 with points ls 1 notitle
    


#fc rgb "cyan" fs pattern 1 bo -1
#set object rect from screen 0.03, screen 0.28 to 0.55, 1 lw 2 fs border lc rgb 'black' #fc rgb "cyan" fs pattern 1 bo -1



#pause -1 "Hit any key to continue"