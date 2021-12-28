package main

import (
	"fmt"
)

// print out arrg vec input from pos position
func print_vec_int(title string, input []int, in_wid int, pos int) {
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

func lRot_int(a []int, rotation int) []int {
	size := len(a)
	var newArray []int
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot_int(a []int, rotation int) []int {
	return lRot_int(a, len(a)-rotation)
}

func addSlice_int(a []int, b []int) []int {
	c := make([]int, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

// same as keep_vec. Only for Test
func keep_vec_int(input []int, in_wid, kp_wid, ul int) []int {
	output := make([]int, len(input))

	tmp := gen_keep_vec(len(input), in_wid, kp_wid, ul)

	for i := range output {
		output[i] = input[i] * tmp[i]
	}

	return output
}

// extend a vector (with in_wid * in_wid elt) to a strided vector (2 in_wid * 2 in_wid) (both are bitreversed) // No padding
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 1020 // 0000 // 3040 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_vec_nosp(input []int, in_wid int, pos int) []int {
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
		output = addSlice_int(output, rRot_int(tmp, rot))
	}

	return output
}

// embed a vector into larger space (in_wid * in_wid) -> (2 in_wid * 2 in_wid) (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 12 00 // 34 00 // 0000 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_sp_novec(input []int, in_wid int, pos int) []int {
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
		mid_out = addSlice_int(mid_out, rRot_int(tmp, rot))
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
		output = addSlice_int(output, rRot_int(tmp, rot))
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
// Assume N -sized input
func extend_full_nhf_int(input []int, in_wid int, pos int, padding, half bool) []int {
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
				output = addSlice_int(output, rRot_int(tmp, rot))
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
				output = addSlice_int(output, rRot_int(tmp, rot))
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
			mid_out = addSlice_int(mid_out, rRot_int(tmp, rot))
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
			output = addSlice_int(output, rRot_int(tmp, rot))
		}
	}

	return output
}

// for float64
// will be combined with int
func extend_full_nhf(input []float64, in_wid int, pos int, padding bool, half bool) []float64 {
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
				output = addSlice(output, rRot(tmp, rot))
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
			tmp := make([]float64, len(input))
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
			tmp := make([]float64, len(input))
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

// returns the idx and rotations for each idx For extend_full
// m_idx is mid index if mid rotation is required
func gen_extend_full_nhf(vec_size int, in_wid int, pos int, padding bool, half bool) (r_idx, m_idx map[int][]int) {
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
func extend_full_int(input []int, in_wid, kp_wid, pos, ul int) []int {
	output := make([]int, len(input))
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
		tmp := make([]int, len(input))
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
		output = addSlice_int(output, rRot_int(tmp, rot))
	}

	return output
}

func comprs_full_fast_int(input []int, in_wid, kp_wid, pos, ul int) []int {
	output_mid := make([]int, len(input))
	output := make([]int, len(input))
	batch := 2 * len(input) / (in_wid * in_wid)
	if kp_wid < in_wid/2 {
		panic("keep width too small. less than in_wid/2")
	}
	pos = int(reverseBits(uint32(pos), 2))
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

	for j := 0; j < 2*min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				if (ul == 0) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = input[idx]
				}
				if (ul == 1) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
					idx := 2*min_wid*in_wid*b + 2*min_wid*j + i + in_wid*min_wid + min_wid
					tmp[idx] = input[idx]
				}
			}
		}
		rot := -j*min_wid + 2*min_wid*min_wid - min_wid
		output_mid = addSlice_int(output_mid, rRot_int(tmp, rot))
	}
	for b := 0; b < batch; b++ {
		tmp := make([]int, len(input))
		for j := 0; j < 2*min_wid; j++ {
			for i := 0; i < min_wid; i++ {
				// if (ul == 0) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) {
				idx := 2*min_wid*in_wid*b + 3*in_wid/2*min_wid + j*min_wid + i
				tmp[idx] = output_mid[idx]
				// }
				// 	if (ul == 1) && (reverseBits(uint32(in_wid/2+j), log_in_wid) < uint32(kp_wid)) && (reverseBits(uint32(min_wid+i), log_in_wid-1) < uint32(kp_wid-in_wid/2)) {
				// 		idx := 2*min_wid*in_wid*b + j*min_wid + i
				// 		tmp[idx] = input[idx]
				// 	}
			}
		}
		rot := -3*b*min_wid*in_wid/2 + pos*min_wid*in_wid/2*batch - 3*min_wid*in_wid/2
		output = addSlice_int(output, rRot_int(tmp, rot))
	}

	return output
}
