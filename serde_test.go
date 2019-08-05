package tdigest

import (
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
)



// add the values [0,n) to a centroid set, equal weights
func simpleTDigest(n int) *TDigest {
	d := NewWithDecay(1000, 0.9, 500)
	for i := 0; i < n; i++ {
		d.Add(float64(i), 1)
	}
	return d
}

func TestMarshalRoundTrip(t *testing.T) {
	testcase := func(in *TDigest) func(*testing.T) {
		return func(t *testing.T) {
			b, err := in.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary err: %v", err)
			}
			out := new(TDigest)
			err = out.UnmarshalBinary(b)
			if err != nil {
				t.Fatalf("UnmarshalBinary err: %v", err)
			}
			if !reflect.DeepEqual(in, out) {
				t.Errorf("marshaling round trip resulted in changes")
				t.Logf("in: %+v", in)
				t.Logf("out: %+v", out)
			}
		}
	}
	t.Run("empty", testcase(New()))
	t.Run("1 value", testcase(simpleTDigest(1)))
	t.Run("1000 values", testcase(simpleTDigest(1000)))

	d := New()
	d.Add(1, 1)
	d.Add(1, 1)
	d.Add(0, 1)
	t.Run("1, 1, 0 input", testcase(d))
}

func TestUnmarshalErrors(t *testing.T) {
	testcase := func(in []byte, wantErr error) func(*testing.T) {
		return func(t *testing.T) {
			have := new(TDigest)
			err := unmarshalBinary(have, in)
			if err != nil {
				if wantErr == nil {
					t.Fatalf("unexpected unmarshal err: %v", err)
				}
				if err.Error() != wantErr.Error() {
					t.Fatalf("wrong error, want=%q, have=%q", wantErr.Error(), err.Error())
				} else {
					return
				}
			} else if wantErr != nil {
				t.Fatalf("expected err=%q, got nil", wantErr.Error())
			}
		}
	}
	t.Run("nil", testcase(
		nil,
		io.ErrUnexpectedEOF,
	))
	t.Run("bad magic", testcase(
		[]byte{
			0x80, 0x0d,
		},
		errors.New("data corruption detected: invalid header magic value 0x0d80"),
	))
	t.Run("incomplete encoding", testcase(
		[]byte{
			0x80, 0x0c,
			0x00,
		},
		io.ErrUnexpectedEOF,
	))
	t.Run("bad encoding", testcase(
		[]byte{
			0x80, 0x0c,
			0xFF, 0xFF, 0xFF, 0xFF,
		},
		errors.New("data corruption detected: invalid encoding version -1"),
	))
	t.Run("incomplete compression", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00,
		},
		io.ErrUnexpectedEOF,
	))
	t.Run("incomplete n", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x00,
		},
		io.ErrUnexpectedEOF,
	))
	t.Run("negative n", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0xFF, 0xFF, 0xFF, 0xFF,
		},
		errors.New("data corruption detected: number of centroids cannot be negative, have -1"),
	))
	t.Run("huge n", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0xFF, 0xFF, 0xFF, 0x7F,
		},
		errors.New("invalid n, cannot be greater than 2^20: 2147483647"),
	))
	t.Run("missing centroids", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
		},
		io.ErrUnexpectedEOF,
	))
	t.Run("partial centroid", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00,
		},
		io.ErrUnexpectedEOF,
	))
	t.Run("decreasing means", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x02, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F,
		},
		errors.New("data corruption detected: centroid 1 has lower mean (1) than preceding centroid 0 (2)"),
	))
	t.Run("nan mean", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		},
		errors.New("data corruption detected: NaN mean not permitted"),
	))
	t.Run("+inf mean", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F,
		},
		errors.New("data corruption detected: Inf mean not permitted"),
	))
	t.Run("-inf mean", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xFF,
		},
		errors.New("data corruption detected: Inf mean not permitted"),
	))
}

func testUnmarshal(t *testing.T) {
	testcase := func(in []byte, want *TDigest) func(*testing.T) {
		return func(t *testing.T) {
			have := new(TDigest)
			err := unmarshalBinary(have, in)
			if err != nil {
				t.Fatalf("unexpected unmarshal err: %v", err)
			}
			if !reflect.DeepEqual(have, want) {
				t.Error("unmarshal did not produce expected digest")
				t.Logf("want=%s", spew.Sprint(want))
				t.Logf("have=%s", spew.Sprint(have))
			}
		}
	}
	t.Run("no centroids", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x00, 0x00, 0x00, 0x00,
		},
		&TDigest{
			processed:   make(CentroidList, 0),
			Compression: 100,
			processedWeight:  0,
		},
	))
	t.Run("one centroid", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F,
		},
		&TDigest{
			processed: []Centroid{
				{
					Weight: 1,
					Mean:  1,
				},
			},
			Compression: 100,
			processedWeight:  1,
		},
	))
	t.Run("two centroids", testcase(
		[]byte{
			0x80, 0x0c,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40,
			0x02, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F,
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
		},
		&TDigest{
			processed: CentroidList{
				{
					Weight: 1,
					Mean:  1,
				},
				{
					Weight: 1,
					Mean:  2,
				},
			},
			Compression: 100,
			processedWeight:  2,
		},
	))
}

