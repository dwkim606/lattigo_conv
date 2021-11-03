package main

import (
	"fmt"
	"math"
)

// distribute input to the output starting from pos position
func arrgvec(input []int, output []int, pos int) {
	batch := len(output) / len(input)

	for i, elt := range input {
		output[pos+i*batch] = elt
	}
}

// print out arrg vec input from pos position
func print_vec(title string, input []int, in_wid int, pos int) {
	row := make([]int, in_wid)
	step := len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid; j++ {
		for i := range row {
			row[i] = input[(j*in_wid+i)*step+pos]

		}
		fmt.Println(row)
	}
	fmt.Println()

}

func print_vec_fl(title string, input []float64, in_wid int, pos int) {
	row := make([]float64, in_wid)
	step := len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid; j++ {
		for i := range row {
			row[i] = input[(j*in_wid+i)*step+pos]
		}
		fmt.Println(row)
	}
	fmt.Println()

}

// print out arrg vec input from pos position (for half vec)
func print_vec_hf(title string, input []int, in_wid int, pos int) {
	row := make([]int, in_wid)
	step := 2 * len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid/2; j++ {
		for i := range row {
			row[i] = input[(j*in_wid+i)*step+pos]

		}
		fmt.Println(row)
	}
	fmt.Println()

}

func lRot(a []int, rotation int) []int {
	size := len(a)
	var newArray []int
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot(a []int, rotation int) []int {
	return lRot(a, len(a)-rotation)
}

func lRot_fl(a []float64, rotation int) []float64 {
	size := len(a)
	var newArray []float64
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot_fl(a []float64, rotation int) []float64 {
	return lRot_fl(a, len(a)-rotation)
}

func addSlice(a []int, b []int) []int {
	c := make([]int, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

func addSlice_fl(a []float64, b []float64) []float64 {
	c := make([]float64, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

// zero padding (extend) the given input
// assume that 4 * len(input) =< N, and
func inputExt(input []float64, logN, in_wid int, print bool) []float64 {
	N := 1 << logN
	in_size := in_wid * in_wid
	batch := len(input) / in_size
	max_batch := N / (4 * in_size)

	sm_input := make([]int, in_size) // each will be packed to tmp vector
	tmp := make([]int, N)
	tmp_rev := make([]int, N)
	test_out := make([]int, N)
	test_out_fl := make([]float64, N)

	// set input and desired output

	for b := 0; b < max_batch; b++ {
		if b < batch {
			for i := range sm_input {
				sm_input[i] = batch*i + b + 1
			}
		} else {
			for i := range sm_input {
				sm_input[i] = 0
			}
		}

		arrgvec(sm_input, tmp, b)
	}

	// if print {
	// 	for b := 0; b < 4; b++ {
	// 		print_vec("input ("+strconv.Itoa(b)+")", tmp, in_wid, b)
	// 	}
	// }
	for i, elt := range tmp {
		tmp_rev[reverseBits(uint32(i), logN)] = elt
	}

	test_out_rev := extend_sp(tmp_rev, in_wid, 0)

	for i, elt := range test_out_rev {
		test_out[reverseBits(uint32(i), logN)] = elt
	}
	// if print {
	// 	for b := 0; b < 1; b++ {
	// 		print_vec("output ("+strconv.Itoa(b)+")", test_out, 2*in_wid, b)
	// 	}
	// }

	for i, elt := range test_out {
		if elt != 0 {
			test_out_fl[i] = input[elt-1]
		}
	}

	return test_out_fl
}

// (bit-reversed) input vector (upper or lower part) has sm vectors (with in_wid * in_wid elt, along with non-valid elements at pad)
// Keep only the valid values, i.e. remove padded values
// e.g., 1* // ** -> 10 // 00 (before bitreversed, pad = 1)
// Only for Test
func keep_vec(input []int, in_wid, pad, pos int) []int {
	bits := int(math.Logb(float64(len(input))))
	output := make([]int, len(input))

	tmp := gen_keep_vec(bits+1, in_wid, pad, pos)

	for i := range output {
		output[i] = input[i] * tmp[i]
	}

	return output
}

// Only for Test
func keep_vec_fl(input []float64, in_wid, pad, pos int) []float64 {
	bits := int(math.Logb(float64(len(input))))
	output := make([]float64, len(input))

	tmp := gen_keep_vec(bits+1, in_wid, pad, pos)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

// returns the idx for keep_vec
// N: length of input (upper + lower)
// pos = 0 -> upper part, pos = 1 -> lower part
func gen_keep_vec(logN, in_wid, pad, pos int) (idx []int) {
	N2 := 1 << (logN - 1)
	idx = make([]int, N2)
	batch := 2 * N2 / (in_wid * in_wid)

	if pos == 0 {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < in_wid-pad; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else if pos == 1 {
		for i := 0; i < in_wid/2-pad; i++ {
			for j := 0; j < in_wid-pad; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else {
		panic(err)
	}

	return idx
}

// extend a vector (with in_wid * in_wid elt) to a strided vector (2 in_wid * 2 in_wid) (both are bitreversed) // No padding
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 1020 // 0000 // 3040 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_vec(input []int, in_wid int, pos int) []int {
	pos = int(reverseBits(uint32(pos), 2))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for l := 0; l < in_wid; l++ {
				idx := 4*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
				tmp[idx] = input[idx]
			}
		}
		rot := j*in_wid - pos*in_wid*in_wid
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// embed a vector into larger space (in_wid * in_wid) -> (2 in_wid * 2 in_wid) (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 12 00 // 34 00 // 0000 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_sp(input []int, in_wid int, pos int) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	pos = int(reverseBits(uint32(pos), 2))
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for i := 0; i < in_wid; i++ {
				idx := 4*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + i
				tmp[idx] = input[idx]
			}
		}
		rot := 3*j*in_wid - pos*in_wid*in_wid
		mid_out = addSlice(mid_out, rRot(tmp, rot))
	}

	for i := 0; i < in_wid; i++ { // kinds of mov depends on i
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for j := 0; j < in_wid; j++ {
				idx := 4*in_wid*in_wid*b + 4*in_wid*j + i
				tmp[idx] = mid_out[idx]
			}
		}
		rot := i
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// extend_vec then extend_sp (both are bitreversed)
// (no padding: with in_wid * in_wid elt -> 4wid * 4wid)
// (padding: in_wid * in_wid (having in/2 * in/2 elts) -> 2wid * 2wid)
// assume that the full vector is filled with sm vectors
// padding = true: sm vector is already inside the 4*len(sm_vector) size vector with zeros
// e.g., 12 00 // 34 00 // 00 00 // 00 00
// 0 <= pos < 4 determines which part of input is extended to output
// padding = false: then, 0 <= pos < 16
// half = true: input vector is of size N/2
func extend_full(input []int, in_wid int, pos int, padding bool, half bool) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)

	if padding {
		pos := int(reverseBits(uint32(pos), 2))
		min_wid := in_wid / 2
		if in_wid%2 != 0 {
			panic("in wid not divisible by 2")
		}

		if half {
			min_batch := batch / 2
			if batch%2 != 0 {
				panic("batch size not divisible by 2")
			}

			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, len(input))
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*min_wid*b + in_wid*min_wid*pos + in_wid*j + i
						tmp[idx] = input[idx]
					}
				}
				rot := in_wid*j - in_wid*min_wid*pos
				output = addSlice(output, rRot(tmp, rot))
			}
		} else {
			min_batch := batch / 4
			if batch%4 != 0 {
				panic("batch size not divisible by 4")
			}

			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, len(input))
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*in_wid*b + in_wid*in_wid*pos + 2*in_wid*j + 2*i
						tmp[idx] = input[idx]
					}
				}
				rot := 2*in_wid*j - in_wid*in_wid*pos
				output = addSlice(output, rRot(tmp, rot))
			}
		}
	} else {
		pos = int(reverseBits(uint32(pos), 4))
		min_batch := batch / 16
		if batch%16 != 0 {
			panic("batch size not divisible by 16")
		}

		for j := 0; j < in_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < min_batch; b++ {
				for l := 0; l < in_wid; l++ {
					idx := 16*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
					tmp[idx] = input[idx]
				}
			}
			rot := 7*j*in_wid - pos*in_wid*in_wid
			mid_out = addSlice(mid_out, rRot(tmp, rot))
		}

		for i := 0; i < in_wid; i++ { // kinds of mov depends on i
			tmp := make([]int, len(input))
			for b := 0; b < min_batch; b++ {
				for j := 0; j < in_wid; j++ {
					idx := 16*in_wid*in_wid*b + 8*in_wid*j + i
					tmp[idx] = mid_out[idx]
				}
			}
			rot := i
			output = addSlice(output, rRot(tmp, rot))
		}
	}

	return output
}

// for float64
// will be combined with int
func extend_full_fl(input []float64, in_wid int, pos int, padding bool, half bool) []float64 {
	mid_out := make([]float64, len(input))
	output := make([]float64, len(input))
	batch := len(input) / (in_wid * in_wid)

	if padding {
		pos := int(reverseBits(uint32(pos), 2))
		min_wid := in_wid / 2
		if in_wid%2 != 0 {
			panic("in wid not divisible by 2")
		}

		if half {
			min_batch := batch / 2
			if batch%2 != 0 {
				panic("batch size not divisible by 2")
			}

			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*min_wid*b + in_wid*min_wid*pos + in_wid*j + i
						tmp[idx] = input[idx]
					}
				}
				rot := in_wid*j - in_wid*min_wid*pos
				output = addSlice_fl(output, rRot_fl(tmp, rot))
			}
		} else {
			min_batch := batch / 4
			if batch%4 != 0 {
				panic("batch size not divisible by 4")
			}

			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*in_wid*b + in_wid*in_wid*pos + 2*in_wid*j + 2*i
						tmp[idx] = input[idx]
					}
				}
				rot := 2*in_wid*j - in_wid*in_wid*pos
				output = addSlice_fl(output, rRot_fl(tmp, rot))
			}
		}
	} else {
		pos = int(reverseBits(uint32(pos), 4))
		min_batch := batch / 16
		if batch%16 != 0 {
			panic("batch size not divisible by 16")
		}

		for j := 0; j < in_wid; j++ { // kinds of mov depends on j
			tmp := make([]float64, len(input))
			for b := 0; b < min_batch; b++ {
				for l := 0; l < in_wid; l++ {
					idx := 16*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
					tmp[idx] = input[idx]
				}
			}
			rot := 7*j*in_wid - pos*in_wid*in_wid
			mid_out = addSlice_fl(mid_out, rRot_fl(tmp, rot))
		}

		for i := 0; i < in_wid; i++ { // kinds of mov depends on i
			tmp := make([]float64, len(input))
			for b := 0; b < min_batch; b++ {
				for j := 0; j < in_wid; j++ {
					idx := 16*in_wid*in_wid*b + 8*in_wid*j + i
					tmp[idx] = mid_out[idx]
				}
			}
			rot := i
			output = addSlice_fl(output, rRot_fl(tmp, rot))
		}
	}

	return output
}

// returns the idx and rotations for each idx For extend_full
// m_idx is mid index if mid rotation is required
func gen_extend_full(vec_size int, in_wid int, pos int, padding bool, half bool) (r_idx, m_idx map[int][]int) {
	r_idx = make(map[int][]int)
	m_idx = make(map[int][]int)
	batch := vec_size / (in_wid * in_wid)

	if padding {
		pos := int(reverseBits(uint32(pos), 2))
		min_wid := in_wid / 2
		if in_wid%2 != 0 {
			panic("in wid not divisible by 2")
		}
		if half {
			min_batch := batch / 2
			if batch%2 != 0 {
				panic("batch size not divisible by 2")
			}
			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*min_wid*b + in_wid*min_wid*pos + in_wid*j + i
						tmp[idx] = 1
					}
				}
				rot := -in_wid*j + in_wid*min_wid*pos
				r_idx[rot] = tmp
				// output = addSlice(output, rRot(tmp, rot))
			}
		} else {
			min_batch := batch / 4
			if batch%4 != 0 {
				panic("batch size not divisible by 4")
			}
			for j := 0; j < min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < min_batch; b++ {
					for i := 0; i < min_wid; i++ {
						idx := 4*in_wid*in_wid*b + in_wid*in_wid*pos + 2*in_wid*j + 2*i
						tmp[idx] = 1
					}
				}
				rot := -2*in_wid*j + in_wid*in_wid*pos
				r_idx[rot] = tmp
				// output = addSlice(output, rRot(tmp, rot))
			}
		}
	} else {
		pos = int(reverseBits(uint32(pos), 4))
		min_batch := batch / 16
		if batch%16 != 0 {
			panic("batch size not divisible by 16")
		}

		for j := 0; j < in_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, vec_size)
			for b := 0; b < min_batch; b++ {
				for l := 0; l < in_wid; l++ {
					idx := 16*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
					tmp[idx] = 1
				}
			}
			rot := -7*j*in_wid + pos*in_wid*in_wid
			m_idx[rot] = tmp
			// mid_out = addSlice(mid_out, rRot(tmp, rot))
		}

		for i := 0; i < in_wid; i++ { // kinds of mov depends on i
			tmp := make([]int, vec_size)
			for b := 0; b < min_batch; b++ {
				for j := 0; j < in_wid; j++ {
					idx := 16*in_wid*in_wid*b + 8*in_wid*j + i
					tmp[idx] = 1
				}
			}
			rot := -i
			r_idx[rot] = tmp
			// output = addSlice(output, rRot(tmp, rot))
		}
	}

	return r_idx, m_idx
}

// reverse of extend_vec / compress a strided  vector (in_wid * in_wid) to (in_wid/2 * in_wid/2) (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1020 // 0000 // 3040 // 0000  -> 1234 (before bitreversed)
// 0 <= pos < 4 determines to which part the output is positioned at the final output
func comprs_vec(input []int, in_wid int, pos int) []int {
	pos = int(reverseBits(uint32(pos), 2))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("input width not divisible by 2")
	}

	for j := 0; j < min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				idx := 4*min_wid*min_wid*b + in_wid*j + i
				tmp[idx] = input[idx]
			}
		}
		rot := -j*min_wid + pos*min_wid*min_wid
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// reverse of extend_full (strided -> normal)
// (output) padding = true: (in_wid * in_wid) -> (in/2 * in/2), result sm vector is inside the 4*len(sm_vector) size vector with zeros
// (output) padding = false:  ;; -> (in/4 * in/4) result sm vector witout padding
// e.g., 12 00 // 34 00 // 00 00 // 00 00
// 0 <= pos < 4 determines to which part the output is positioned at the final output
// padding = false: then, 0 <= pos < 16, and result sm vector has no zeros
func comprs_full(input []int, in_wid int, pos int, padding bool) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)

	if padding {
		pos = int(reverseBits(uint32(pos), 2))
		min_wid := in_wid / 4
		if in_wid%4 != 0 {
			panic("input wid not divisible by 4")
		}

		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := in_wid*in_wid*b + 2*in_wid*j + 2*i
					tmp[idx] = input[idx]
				}
			}
			rot := -4*j*min_wid + 4*pos*min_wid*min_wid
			output = addSlice(output, rRot(tmp, rot))
		}
	} else {
		pos = int(reverseBits(uint32(pos), 4))
		min_wid := in_wid / 4
		if in_wid%4 != 0 {
			panic("in width not divisible by 4")
		}

		for i := 0; i < min_wid; i++ { // kinds of mov depends on i
			tmp := make([]int, len(input))
			for b := 0; b < batch; b++ {
				for j := 0; j < min_wid; j++ {
					idx := in_wid*in_wid*b + 2*in_wid*j + 2*i
					tmp[idx] = input[idx]
				}
			}
			rot := -i
			mid_out = addSlice(mid_out, rRot(tmp, rot))
		}

		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := in_wid*in_wid*b + 2*in_wid*j + i
					tmp[idx] = mid_out[idx]
				}
			}
			rot := -7*j*min_wid + pos*min_wid*min_wid
			output = addSlice(output, rRot(tmp, rot))
		}
	}

	return output
}
