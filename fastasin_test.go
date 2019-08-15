package tdigest

import (
	"fmt"
	"github.com/coreos/etcd/pkg/testutil"
	"math"
	"testing"
)

func TestFastAsin(t *testing.T) {
	tests := []struct {
		name    string
		in      float64
		outTest func(float64) bool
	}{
		{"neg", -1, func(x float64) bool { return x == -1.5707963267948966 }},
		{"nan", 4, math.IsNaN},
		{"neg", 0.9, func(x float64) bool { return x == 1.1197695149986342 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fmt.Println(fastAsin(test.in))
			testutil.AssertTrue(t, test.outTest(fastAsin(test.in)))
		})
	}
}
