### horizontal stacked histogram
set terminal eps
set output  "boot_graph_row.eps"   
reset session

$Data <<EOD
Type Conv	ReLU	StoC	Sine+CtoS
BL   2.61	4.49	8.47	45.31
Ours 0.22	5.43	1.06	22.77
BL   22.31 4.51 8.5 45.21
Ours 0.9	5.43	1.06	22.8
BL 157.24	4.46	8.22	44.7
Ours 3.61	5.39	1.04	22.49
EOD

set ylabel "BL vs Ours (kernel width, # batch)"
set xlabel "Running Time (seconds)"
# set xrange [0:103]
set yrange [:] reverse
set offsets 0,0,0.5,0.5
set style fill solid 0.75
set key out

ColCount = 4
myBoxwidth = 0.5

plot for [col=2:ColCount+1] $Data u col:0: \
 (total=(sum [i=2:ColCount+1] column(i)),(sum [i=2:col-1] column(i))): \
 ((sum [i=2:col] column(i))):($0-myBoxwidth/2):($0+myBoxwidth/2):ytic(1) w boxxyerror ti columnhead(col)
### end of code

