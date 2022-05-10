set terminal eps       
set output  "conv_graph.eps"   
set multiplot

set object rect from screen 0.01, screen 0.05 to screen 0.475, screen 0.2 dashtype 2 lw 2 fs empty border lc rgb 'black' #fc rgb "cyan" fs pattern 1 bo -1
set object rect from 4,100 to 110,625 lw 1 fs border lc rgb 'black'
set arrow from screen 0.01, screen 0.2 to 4,100 dashtype 2 lw 2 lc rgb 'black' nohead
set arrow from screen 0.475, screen 0.2 to 110,100 dashtype 2 lw 2 lc rgb 'black' nohead
set arrow from 4,32 to 1024, 32 dt 4 lw 3 lc rgb "#2E8957" nohead
set label "Boot" at 625, 48 tc rgb "#2E8957"


set xlabel "Number of Batches"
set y2label "Running Time (seconds)"
set logscale x 4
# set logscale y
set xrange [4:1024]
set xtics (4,16,64,256,1024)
set y2tics mirror
set link y2 via y inv y
set my2tics
unset ytics


set grid xtics
# set grid y2tics my2tics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 2
set key at 700, 600
set key spacing 1.25


# set style line 1 lc rgb '#48494B' lt 1 lw 3 pt 7 ps 0.5 # gray
# set style line 2 lc rgb '#828282' lt 1 lw 3 pt 7 ps 0.5  # green
# set style line 3 lc rgb '#97978F' lt 1 lw 3 pt 7 ps 0.5 # blue
set style line 1 lc rgb '#0A1172' lt 1 lw 3 pt 7 ps 0.5 # gray
set style line 2 lc rgb '#0147AB' lt 1 lw 3 pt 7 ps 0.5  # green
set style line 3 lc rgb '#588BAE' lt 1 lw 3 pt 7 ps 0.5 # blue
set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # blue
# set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # orange


plot "conv.dat" using 1:4 w l smooth bezier ls 1 notitle, \
    "conv.dat" using 1:4 with points ls 1 notitle "Ker7", \
    1/0 with linespoints ls 1 title "Ker7", \
    "conv.dat" using 1:3 smooth bezier ls 2 notitle, \
    "conv.dat" using 1:3 with points ls 2 notitle, \
    1/0 with linespoints ls 2 title "Ker5", \
    "conv.dat" using 1:2 smooth bezier ls 3 notitle, \
    "conv.dat" using 1:2 with points ls 3 notitle, \
    1/0 with linespoints ls 3 title "Ker3", \
    "conv.dat" using 1:5 smooth bezier ls 4 notitle, \
    "conv.dat" using 1:5 with points ls 4 notitle, \
    1/0 with linespoints ls 4 title "Ours"

#fc rgb "cyan" fs pattern 1 bo -1
#set object rect from screen 0.03, screen 0.28 to 0.55, 1 lw 2 fs border lc rgb 'black' #fc rgb "cyan" fs pattern 1 bo -1

set origin .04, .275
set size .425,.65
clear
unset key
unset grid
unset object
unset arrow
unset label
unset logscale x
unset xlabel
unset y2label

set arrow from 4,32 to 64, 32 dt 4 lw 4 lc rgb "#2E8957" nohead
set label "Boot" at 6, 34 tc rgb "#2E8957"
set logscale x 4
set xrange [4:64]
set xtics (4,16,64)
set grid xtics
set grid y2tics my2tics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 2
set tmargin 1
set lmargin 1
set rmargin 1
plot "conv.dat" using 1:4 w l smooth bezier ls 1 notitle, \
    "conv.dat" using 1:4 with points ls 1 notitle "Ker7", \
    1/0 with linespoints ls 1 title "Ker7", \
    "conv.dat" using 1:3 smooth bezier ls 2 notitle, \
    "conv.dat" using 1:3 with points ls 2 notitle, \
    1/0 with linespoints ls 2 title "Ker5", \
    "conv.dat" using 1:2 smooth bezier ls 3 notitle, \
    "conv.dat" using 1:2 with points ls 3 notitle, \
    1/0 with linespoints ls 3 title "Ker3", \
    "conv.dat" using 1:5 smooth bezier ls 4 notitle, \
    "conv.dat" using 1:5 with points ls 4 notitle, \
    1/0 with linespoints ls 4 title "Ours"
unset multiplot


#pause -1 "Hit any key to continue"