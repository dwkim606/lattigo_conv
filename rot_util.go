package main

import (
	"fmt"
	"math"
)

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(math.Round(num*output)) / output
}

// distribute input to the output starting from pos position
func arrgvec(input []int, output []int, pos int) {
	batch := len(output) / len(input)
	for i, elt := range input {
		output[pos+i*batch] = elt
	}
}

func print_vec(title string, input []float64, in_wid int, pos int) {
	row := make([]float64, in_wid)
	step := len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid; j++ {
		for i := range row {
			row[i] = toFixed(input[(j*in_wid+i)*step+pos], 3)
		}
		fmt.Println(row)
	}
	fmt.Println()

}

func lRot(a []float64, rotation int) []float64 {
	size := len(a)
	var newArray []float64
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot(a []float64, rotation int) []float64 {
	return lRot(a, len(a)-rotation)
}

func addSlice(a []float64, b []float64) []float64 {
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

	for i, elt := range tmp {
		tmp_rev[reverseBits(uint32(i), logN)] = elt
	}

	test_out_rev := extend_sp_novec(tmp_rev, in_wid, 0)

	for i, elt := range test_out_rev {
		test_out[reverseBits(uint32(i), logN)] = elt
	}

	for i, elt := range test_out {
		if elt != 0 {
			test_out_fl[i] = input[elt-1]
		}
	}

	return test_out_fl
}

func gen_expand_BL(vec_size, in_wid int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := vec_size / (in_wid * in_wid)

	if batch%4 != 0 {
		panic("batch not divisible by 4")
	}
	min_batch := batch / 4

	for i := 1; i <= in_wid; i++ { // kinds of mov depends on i
		tmp := make([]int, vec_size)
		for j := 1; j <= in_wid; j++ {
			for b := 0; b < min_batch; b++ {
				idx := (i-1)*in_wid + j - 1 + 4*b*in_wid*in_wid
				tmp[idx] = 1
			}
		}
		rot := in_wid - 3*in_wid*i
		m_idx[rot] = tmp
	}

	for j := 1; j <= in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, vec_size)
		for i := 1; i <= in_wid; i++ {
			for b := 0; b < min_batch; b++ {
				idx := (2*i-1)*2*in_wid + j - 1 + 4*b*in_wid*in_wid
				tmp[idx] = 1
			}
		}
		rot := -j
		r_idx[rot] = tmp
	}

	return m_idx, r_idx
}

func gen_comprs_BL(vec_size, in_wid int) (m_idx, r_idx map[int][]int) {
	m_idx = make(map[int][]int)
	r_idx = make(map[int][]int)
	batch := vec_size / (in_wid * in_wid)

	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("in wid not divisible by 2")
	}

	for j := 1; j <= min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, vec_size)
		for i := 1; i <= min_wid; i++ {
			for b := 0; b < batch; b++ {
				idx := (2*i-1)*in_wid + 2*j - 1 + b*in_wid*in_wid
				tmp[idx] = 1
			}
		}
		rot := j
		m_idx[rot] = tmp
	}

	for i := 1; i <= min_wid; i++ { // kinds of mov depends on i
		tmp := make([]int, vec_size)
		for j := 1; j <= min_wid; j++ {
			for b := 0; b < batch; b++ {
				idx := (2*i-1)*in_wid + j - 1 + b*in_wid*in_wid
				tmp[idx] = 1
			}
		}
		rot := 3*min_wid*i - min_wid
		r_idx[rot] = tmp
	}

	return m_idx, r_idx
}

// (bit-reversed) input vector := (upper or lower part) of the total vector having in_wid * in_wid size elts
// Keep only the kp_wid*kp_wid values
// e.g., 1* // ** -> 10 // 00 (before bitreversed, pad = 1)
// ul: up or low
// assume N/2 sized input
func keep_vec(input []float64, in_wid, kp_wid, ul int) []float64 {
	output := make([]float64, len(input))

	tmp := gen_keep_vec(len(input), in_wid, kp_wid, ul)

	for i := range output {
		output[i] = input[i] * float64(tmp[i])
	}

	return output
}

// returns the idx for keep_vec
// N: length of input (upper + lower)
// ul = 0 -> upper part, ul = 1 -> lower part
func gen_keep_vec(vec_size, in_wid, kp_wid, ul int) (idx []int) {
	logN := 0
	for ; (1 << logN) < (2 * vec_size); logN++ {
	}
	idx = make([]int, vec_size)
	batch := 2 * vec_size / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}

	if ul == 0 {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else if ul == 1 {
		for i := 0; i < kp_wid-in_wid/2; i++ {
			for j := 0; j < kp_wid; j++ {
				for b := 0; b < batch; b++ {
					id := int(reverseBits(uint32(in_wid*batch*i+batch*j+b), logN-1))
					idx[id] = 1
				}
			}
		}
	} else {
		panic("ul not 0 nor 1")
	}

	return idx
}

// Assume N/2 sized input
// extend_vec then extend_sp (both are bitreversed)
// (no padding: with in_wid * in_wid elt -> 4wid * 4wid)
// (padding: in_wid * in_wid (having in/2 * in/2 elts) -> 2wid * 2wid)
// assume that the full vector is filled with sm vectors
// padding = true: sm vector is already inside the 4*len(sm_vector) size vector with zeros
// e.g., 12 00 // 34 00 // 00 00 // 00 00
// 0 <= pos < 4 determines which part of input is extended to output
// padding = false: then, 0 <= pos < 16
// half = true: input vector is of size N/2
func extend_full(input []float64, in_wid, kp_wid, pos, ul int) []float64 {
	output := make([]float64, len(input))
	batch := 2 * len(input) / (in_wid * in_wid)
	pos_ := int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("in wid not divisible by 2")
	}
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]float64, len(input))
		for b := 0; b < min_batch; b++ {
			for i := 0; i < min_wid; i++ {
				// fmt.Println(reverseBits(uint32(j), log_in_wid))
				if (ul == 0) && (reverseBits(uint32(j), log_in_wid) < uint32(kp_wid)) || (ul == 1) && (reverseBits(uint32(j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-1) < uint32(kp_wid-min_wid)) {
					idx := 4*in_wid*min_wid*b + in_wid*min_wid*pos_ + min_wid*j + i
					tmp[idx] = input[idx]
				}
			}
		}
		rot := in_wid*min_wid*2 + min_wid + min_wid*j - in_wid*min_wid*pos_
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// Assume N/2 sized input
// Generate vectors for extend_full
func gen_extend_full(vec_size int, in_wid, kp_wid, pos, ul int) (r_idx map[int][]int) {
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid)
	pos_ := int(reverseBits(uint32(pos), 2))
	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("in wid not divisible by 2")
	}
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, vec_size)
		for b := 0; b < min_batch; b++ {
			for i := 0; i < min_wid; i++ {
				// fmt.Println(reverseBits(uint32(j), log_in_wid))
				if (ul == 0) && (reverseBits(uint32(j), log_in_wid) < uint32(kp_wid)) || (ul == 1) && (reverseBits(uint32(j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(i), log_in_wid-1) < uint32(kp_wid-min_wid)) {
					idx := 4*in_wid*min_wid*b + in_wid*min_wid*pos_ + min_wid*j + i
					tmp[idx] = 1
				}
			}
		}
		rot := -in_wid*min_wid*2 - min_wid - min_wid*j + in_wid*min_wid*pos_
		r_idx[rot] = tmp
	}

	return r_idx
}

// Assume N/2 input vector
// reverse of extend_full (after strided conv -> normal)
// in_wid = input wid including padding
// kp_wid = keep wid
// padding = true: only keeps valid elements	// (output) e.g., 12 00 // 34 00 // 00 00 // 00 00
// padding = false: keeps all elements	// (output) e.g., 12 // 34
// 0 <= pos < 4 determines to which part the output is positioned at the final output
// ul : up (0) or low (1) part
func comprs_full(input []float64, in_wid, kp_wid, pos, ul int) []float64 {
	output := make([]float64, len(input))
	batch := 2 * len(input) / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	padding := false
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if padding {
		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]float64, len(input))
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := 2*min_wid*in_wid*b + in_wid*j + i + min_wid*in_wid + min_wid
					tmp[idx] = input[idx]
				}
			}
			rot := -2*j*min_wid + 2*pos*min_wid*min_wid - min_wid*in_wid - min_wid
			output = addSlice(output, rRot(tmp, rot))
		}
		// // when we want to extract even positioned inputs
		// for j := 0; j < min_wid; j++ { // kinds of mov depends on j
		// 	tmp := make([]int, len(input))
		// 	for b := 0; b < batch; b++ {
		// 		for i := 0; i < min_wid; i++ {
		// 			idx := 2*min_wid*in_wid*b + in_wid*j + i
		// 			tmp[idx] = input[idx]
		// 		}
		// 	}
		// 	rot := -2*j*min_wid + 2*pos*min_wid*min_wid
		// 	output = addSlice(output, rRot(tmp, rot))
		// }
	} else {
		if ul == 0 {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = input[idx]
						}
					}
				}
				rot := -j*min_wid + 2*pos*min_wid*min_wid - min_wid - in_wid*min_wid
				output = addSlice(output, rRot(tmp, rot))
			}
		} else {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]float64, len(input))
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(3*min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = input[idx]
						}
					}
				}
				rot := -j*min_wid + 2*pos*min_wid*min_wid - min_wid - in_wid*min_wid
				output = addSlice(output, rRot(tmp, rot))
			}
		}
		// // when we want to extract even positioned inputs
		// for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		// 	tmp := make([]int, len(input))
		// 	for b := 0; b < batch; b++ {
		// 		for i := 0; i < min_wid; i++ {
		// 			idx := 2*min_wid*in_wid*b + 2*min_wid*j + i
		// 			tmp[idx] = input[idx]
		// 		}
		// 	}
		// 	rot := -j*min_wid + 2*pos*min_wid*min_wid
		// 	output = addSlice(output, rRot(tmp, rot))
		// }
	}

	return output
}

// generate vectors for comprs_full (N/2 input)
// returns the idx and rotations for each idx For comprs_full_hf
// vec_size = slots, in_wid = real in_wid including padding,
// CAUTION: rotation = -rotation (of comprs_full_hf)
func gen_comprs_full(vec_size, in_wid, kp_wid, pos, ul int) (r_idx map[int][]int) {
	r_idx = make(map[int][]int)
	batch := 2 * vec_size / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
	padding := false
	min_wid := in_wid / 4
	if in_wid%4 != 0 {
		panic("input wid not divisible by 4")
	}
	if in_wid%2 != 0 {
		panic("input wid not divisible by 2")
	}
	log_in_wid := 0
	for ; (1 << log_in_wid) < in_wid; log_in_wid++ {
	}

	if padding {
		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, vec_size)
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := 2*min_wid*in_wid*b + in_wid*j + i + min_wid*in_wid + min_wid
					tmp[idx] = 1
				}
			}
			rot := 2*j*min_wid - 2*pos*min_wid*min_wid + min_wid*in_wid + min_wid
			r_idx[rot] = tmp
		}
	} else {
		if ul == 0 {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					if reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid) {
						for i := 0; i < min_wid; i++ {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = 1
						}
					}
				}
				rot := j*min_wid - 2*pos*min_wid*min_wid + min_wid + in_wid*min_wid
				r_idx[rot] = tmp
			}
		} else {
			for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
				tmp := make([]int, vec_size)
				for b := 0; b < batch; b++ {
					for i := 0; i < min_wid; i++ {
						if (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(3*min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
							idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
							tmp[idx] = 1
						}
					}
				}
				rot := j*min_wid - 2*pos*min_wid*min_wid + min_wid + in_wid*min_wid
				r_idx[rot] = tmp
			}
		}
	}

	return r_idx
}
