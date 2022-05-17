set terminal eps       
set output  "search_CNN.eps"   

# set object rect from screen 0.01, screen 0.05 to screen 0.475, screen 0.2 dashtype 2 lw 2 fs empty border lc rgb 'black' #fc rgb "cyan" fs pattern 1 bo -1
# set object rect from 4,100 to 110,625 lw 1 fs border lc rgb 'black'
# set arrow from screen 0.01, screen 0.2 to 4,100 dashtype 2 lw 2 lc rgb 'black' nohead
# set arrow from screen 0.475, screen 0.2 to 110,100 dashtype 2 lw 2 lc rgb 'black' nohead
# set arrow from 4,32 to 1024, 26.82 dt 4 lw 3 lc rgb "#2E8957" nohead
# set label "Bootstrapping" at screen 0.85, screen 0.71 tc rgb "#2E8957"
# set label "Bootstrapping" at screen 0.8, 40 tc rgb "#2E8957"
set multiplot layout 1,3 #title "CNN search on variants of Plain-Resnet20" font ",14"

set xlabel "Number of Layers"
set ylabel "Accuracy"
set xtics (8, 14, 20)
set xrange [7:21]
set yrange [86:95]
set mytics 
set grid xtics
set grid ytics mytics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 3


# set style line 1 lc rgb '#48494B' lt 1 lw 3 pt 7 ps 0.5 # gray
# set style line 2 lc rgb '#828282' lt 1 lw 3 pt 7 ps 0.5  # green
# set style line 3 lc rgb '#97978F' lt 1 lw 3 pt 7 ps 0.5 # blue
set style line 1 lc rgb '#97978F' lt 1 lw 3 pt 7 ps 0.5 # gray #'#0A1172'
set style line 2 lc rgb '#028A0F'lt 1 lw 3 pt 7 ps 0.5  # green
set style line 3 lc rgb '#ed7014' lt 1 lw 3 pt 7 ps 0.5 # blue
# set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # blue
# set style line 4 lc rgb '#ed7014' lt 1 lw 3 pt 9 ps 1.0 # orange

set title "Wide1 (Original)"
set key left top
set key spacing 1.25 box
plot "search.dat" using 1:2:3:4 with yerr notitle ls 3, \
    "search.dat" using 1:2 with linespoints ls 3 title "Ker3", \
    "search.dat" using 1:5:6:7 with yerr notitle ls 2, \
    "search.dat" using 1:5 with linespoints ls 2 title "Ker5", \
    "search.dat" using 1:8:9:10 with yerr notitle ls 1, \
    "search.dat" using 1:8 with linespoints ls 1 title "Ker7"

set title "Wide2 variant"
unset key
plot "search2.dat" using 1:2:3:4 with yerr notitle ls 3, \
    "search2.dat" using 1:2 with linespoints ls 3 title "Ker3", \
    "search2.dat" using 1:5:6:7 with yerr notitle ls 2, \
    "search2.dat" using 1:5 with linespoints ls 2 title "Ker5", \
    "search2.dat" using 1:8:9:10 with yerr notitle ls 1, \
    "search2.dat" using 1:8 with linespoints ls 1 title "Ker7"

set title "Wide3 variant"
unset key
plot "search3.dat" using 1:2:3:4 with yerr notitle ls 3, \
    "search3.dat" using 1:2 with linespoints ls 3 title "Ker3", \
    "search3.dat" using 1:5:6:7 with yerr notitle ls 2, \
    "search3.dat" using 1:5 with linespoints ls 2 title "Ker5", \
    "search3.dat" using 1:8:9:10 with yerr notitle ls 1, \
    "search3.dat" using 1:8 with linespoints ls 1 title "Ker7"

unset multiplot





#fc rgb "cyan" fs pattern 1 bo -1
#set object rect from screen 0.03, screen 0.28 to 0.55, 1 lw 2 fs border lc rgb 'black' #fc rgb "cyan" fs pattern 1 bo -1

# set origin .04, .275
# set size .425,.65
# clear
# unset key
# unset grid
# unset object
# unset arrow
# unset label
# unset logscale x
# unset xlabel
# unset y2label

# set arrow from 4,32 to 64, 32 dt 4 lw 4 lc rgb "#2E8957" nohead
# set label "Boot" at 6, 34 tc rgb "#2E8957"
# set logscale x 4
# set xrange [4:64]
# set xtics (4,16,64)
# set grid xtics
# set grid y2tics my2tics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 2
# set tmargin 1
# set lmargin 1
# set rmargin 1
# plot "search.dat" using 1:4 w l smooth bezier ls 1 notitle, \
#     "search.dat" using 1:4 with points ls 1 notitle "Ker7", \
#     1/0 with linespoints ls 1 title "Ker7", \
#     "search.dat" using 1:3 smooth bezier ls 2 notitle, \
#     "search.dat" using 1:3 with points ls 2 notitle, \
#     1/0 with linespoints ls 2 title "Ker5", \
#     "search.dat" using 1:2 smooth bezier ls 3 notitle, \
#     "search.dat" using 1:2 with points ls 3 notitle, \
#     1/0 with linespoints ls 3 title "Ker3", \
#     "search.dat" using 1:5 smooth bezier ls 4 notitle, \
#     "search.dat" using 1:5 with points ls 4 notitle, \
#     1/0 with linespoints ls 4 title "Ours"
# unset multiplot


#pause -1 "Hit any key to continue"