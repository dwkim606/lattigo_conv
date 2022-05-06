set terminal eps size 4, 6
set output  "boot_graph.eps"   

set style data histogram
set style histogram rowstacked title offset -0.5, -1.5
set style fill solid
set boxwidth 0.75
# set key invert samplen 0.2
# set key samplen 0.2
# set ytics rotate 90
set xtics rotate 90
set y2label "Running Time (seconds)" rotate by 90
set y2tics mirror rotate by 90 offset 0, -1
set link y2 via y inv y
set my2tics
set y2range [:250]
unset ytics
set grid y2tics my2tics lc rgb 'gray' lt 1 lw 2, lc rgb 'gray' dt 3 lt 3 lw 3

unset key
# set key invert left top
set lmargin 6
set bmargin 7

x = 0.035
y = 0.2
Dy = 0.2

rx = 0.015
ry = 0.26
dx = 0.05
dy = 0.04

set label 'Conv' at screen x, screen y rotate by 90
set object rect from screen rx, screen ry to screen rx+dx, screen ry+dy fc rgbcolor "#ED7117" fs pattern 1

set label 'ReLU' at screen x, screen y+Dy rotate by 90
set object rect from screen rx, screen ry+Dy to screen rx+dx, screen ry+Dy+dy fc rgbcolor "#B3DFE9"

set label 'StoC' at screen x, screen y+2*Dy rotate by 90
set object rect from screen rx, screen ry+2*Dy to screen rx+dx, screen ry+2*Dy+dy fc rgbcolor "#234F1E" fs pattern 10

set label 'CtoS+Sine' at screen x, screen y+3*Dy rotate by 90
set object rect from screen rx, screen ry+3*Dy+0.05 to screen rx+dx, screen ry+3*Dy+0.05+dy fc rgbcolor '#DEDEDE'

plot newhistogram "(Ker3, B16)" rotate by 135 lt 1, \
     'boot.dat' index 0 u 2:xtic(1) title "Conv" lc rgbcolor "#ED7117" fs pattern 1 border -1, \
     '' index 0 u 3 title "ReLU" lc rgbcolor "#B3DFE9" fs border -1, \
     '' index 0 u 4 title "StoC" lc rgbcolor "#234F1E" fs pattern 10 border -1, \
     '' index 0 u 5 title "CtoS+Sine" lc rgbcolor "#DEDEDE" fs border -1, \
     newhistogram "(Ker5, B64)" rotate by 135 lt 1, \
     'boot.dat' index 1 u 2:xtic(1) notitle lc rgbcolor "#ED7117" fs pattern 1 border -1, \
     '' index 1 u 3 notitle lc rgbcolor "#B3DFE9" fs border -1, \
     '' index 1 u 4 notitle lc rgbcolor "#234F1E" fs pattern 10 border -1, \
     '' index 1 u 5 notitle lc rgbcolor "#DEDEDE" fs border -1,\
      newhistogram "(Ker7, B256)" rotate by 135 lt 1, \
     'boot.dat' index 2 u 2:xtic(1) notitle lc rgbcolor "#ED7117" fs pattern 1 border -1, \
     '' index 2 u 3 notitle lc rgbcolor "#B3DFE9" fs border -1, \
     '' index 2 u 4 notitle lc rgbcolor "#234F1E" fs pattern 10 border -1,  \
     '' index 2 u 5 notitle lc rgbcolor "#DEDEDE" fs border -1


