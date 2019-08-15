package tdigest

import (
	"testing"

	"fmt"
	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"reflect"
	"time"
	"strconv"
)

const (
	N     = 1e6
	Mu    = 10
	Sigma = 3

	seed = 42
)

var (
	// NormalData is a slice of N random values that are normaly distributed with mean Mu and standard deviation Sigma.
	NormalData           []float64
	UniformData          []float64
	benchmarkCompression = float64(50)
	benchmarkMedian      = float64(2.675264e09)
	benchmarkStdDev      = float64(13.14254e09)
	benchmarkDecayValue  = 0.9
	benchmarkDecayEvery  = int32(1000)
	benchmarks           = []struct {
		name  string
		scale scaler
	}{
		{name: "k1", scale: &K1{}},
		{name: "k1_fast", scale: &K1Fast{}},
		{name: "k1_spliced", scale: &K1Spliced{}},
		{name: "k1_spliced_fast", scale: &K1SplicedFast{}},
		{name: "k2", scale: &K2{}},
		{name: "k2_spliced", scale: &K2Spliced{}},
		//{name: "k3", scale: &K3{}},
		{name: "k3_spliced", scale: &K3Spliced{}},
		{name: "kquadratic", scale: &KQuadratic{}},
	}
)

func init() {
	dist := distuv.Normal{
		Mu:    Mu,
		Sigma: Sigma,
		Src:   rand.New(rand.NewSource(seed)),
	}
	uniform := rand.New(rand.NewSource(seed))

	UniformData = make([]float64, N)

	NormalData = make([]float64, N)

	for i := range NormalData {
		NormalData[i] = dist.Rand()
		UniformData[i] = uniform.Float64() * 100
	}
}

func TestTdigest_Quantile(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		quantile float64
		want     float64
		epsilon  float64
	}{
		{name: "increasing", quantile: 0.5, data: []float64{1, 2, 3, 4, 5}, want: 3},
		{name: "data in decreasing order", quantile: 0.25, data: []float64{555.349107, 432.842597}, want: 432.842597},
		{name: "small", quantile: 0.5, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 3},
		{name: "small 99 (max)", quantile: 0.99, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 5},
		{name: "normal 50", quantile: 0.5, data: NormalData, want: 10.000744215323294, epsilon: 0.000124},
		{name: "normal 90", quantile: 0.9, data: NormalData, want: 13.841895725158281, epsilon: 1.6e-05},
		{name: "uniform 50", quantile: 0.5, data: UniformData, want: 49.992136904768316, epsilon: 0.000207},
		{name: "uniform 90", quantile: 0.9, data: UniformData, want: 89.98220402280788, epsilon: 4.6e-05},
		{name: "uniform 99", quantile: 0.99, data: UniformData, want: 98.98511738020078, epsilon: 1.1e-05},
		{name: "uniform 99.9", quantile: 0.999, data: UniformData, want: 99.90131708898765, epsilon: 2.52e-06},
	}
	for _, tt := range tests {
		for _, bt := range benchmarks {
			t.Run(tt.name+"-"+bt.name, func(t *testing.T) {
				td := NewWithCompression(1000)
				td.Scaler = bt.scale
				for _, x := range tt.data {
					td.Add(x, 1)
				}
				got := td.Quantile(tt.quantile)
				actual := quantile(tt.quantile, tt.data)
				assert.InEpsilon(t, tt.want, got, tt.epsilon, "unexpected quantile %f, got %g want %g", tt.quantile, got, tt.want)
				assert.InEpsilon(t, actual, got, tt.epsilon, "unexpected quantile %f, got %g want %g", tt.quantile, got, tt.want)
			})
		}
	}
}

func TestClone(t *testing.T) {
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

func TestTdigest_CDFs(t *testing.T) {
	tests := []struct {
		name    string
		data    []float64
		cdf     float64
		want    float64
		epsilon float64
	}{
		{name: "increasing", cdf: 3, data: []float64{1, 2, 3, 4, 5}, want: 0.5},
		{name: "small", cdf: 4, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 0.7, epsilon: 0.072},
		{name: "small max", cdf: 5, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 0.9, epsilon: 0.12},
		{name: "normal mean", cdf: 10, data: NormalData, want: 0.499925, epsilon: 2.2e-05},
		{name: "normal high", cdf: -100, data: NormalData, want: 0},
		{name: "normal low", cdf: 110, data: NormalData, want: 1},
		{name: "uniform 50", cdf: 50, data: UniformData, want: 0.500068, epsilon: 0.000116},
		{name: "uniform min", cdf: 0, data: UniformData, want: 0},
		{name: "uniform max", cdf: 100, data: UniformData, want: 1},
		{name: "uniform 10", cdf: 10, data: UniformData, want: 0.099872, epsilon: 0.000158},
		{name: "uniform 90", cdf: 90, data: UniformData, want: 0.900155, epsilon: 3.3e-5},
	}
	for _, tt := range tests {
		for _, bt := range benchmarks {
			t.Run(tt.name+"-"+bt.name, func(t *testing.T) {
				td := NewWithCompression(1000)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
				got := td.CDF(tt.cdf)
				actual := cdf(tt.cdf, tt.data)
				if got != tt.want {
					assert.InEpsilon(t, tt.want, got, tt.epsilon, "unexpected CDF %f, got %g want %g", tt.cdf, got, tt.want)
				}
				if got != actual {
					assert.InEpsilon(t, actual, got, tt.epsilon, "unexpected CDF %f, got %g want %g", tt.cdf, got, tt.want)
				}
			})
		}
	}
}

func TestCloneRoundTrip(t *testing.T) {
	testcase := func(in *TDigest) func(*testing.T) {
		return func(t *testing.T) {

			out := in.Clone()
			if !reflect.DeepEqual(in, out) {
				t.Errorf("marshaling round trip resulted in changes")
				t.Logf("inn: %+v", in)
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

func getVal() float64 {
	return math.Abs(rand.NormFloat64())*benchmarkStdDev + benchmarkMedian
}

func TestSizesVsCap(t *testing.T) {
	m := map[string]int{
		"k1":              312,
		"k1_fast":         314,
		"k1_spliced":      252,
		"k1_spliced_fast": 253,
		"k2":              325,
		"k2_spliced":      162,
		"k3_spliced":      175,
		"kquadratic":      306,
	}
	for _, test := range benchmarks {
		t.Run(test.name, func(t *testing.T) {
			td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			td.Scaler = test.scale
			n := 1000000
			for i := 0; i < n; i++ {
				td.Add(getVal(), 1.0)
			}
			td.process()
			fmt.Printf("\t\t\t\t\t\tn: %d len: %d cap: %d\n", n, len(td.processed), cap(td.processed))

			if len(td.processed) > m[test.name] {
				t.Errorf("unexpected centroid size %d > %d", len(td.processed), m[test.name])
			}
		})
	}
}

func BenchmarkMainAdd(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	b.ReportAllocs()
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			_ = getTd(bm, b)
		})
	}
}

func getTd(bm struct {
	name  string
	scale scaler
}, b *testing.B) *TDigest {
	td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
	data := getData(b)
	td.Scaler = bm.scale
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		td.Add(data[i], 1.0)
	}
	return td
}

func BenchmarkMainQuantile(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			td := getTd(bm, b)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				td.Quantile(rand.Float64())
			}
		})
	}
}

func BenchmarkMainCDF(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			td := getTd(bm, b)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				td.CDF(getVal())
			}
		})
	}
}

func BenchmarkCompression(b *testing.B) {
	benchmarks := []struct {
		compression int
	}{
		{1000},
		{500},
		{250},
		{125},
		{100},
		{50},
	}
	for _, bm := range benchmarks {
		b.Run("Compression "+strconv.Itoa(bm.compression), func(b *testing.B) {
			b.ReportAllocs()
			td := NewWithDecay(float64(bm.compression), benchmarkDecayValue, benchmarkDecayEvery)
			data := make([]float64, b.N)
			for i := 0; i < b.N; i++ {
				data[i] = getVal()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1.0)
			}
			q := td.Quantile(0.99)
			b.StopTimer()
			actual := quantile(0.99, data)
			fmt.Println("\n", "proc", len(td.processed), cap(td.processed), "unproc", cap(td.unprocessed), td.maxProcessed, q, actual, math.Abs(q-actual))
		})
	}
}

func BenchmarkMultipleHistos(b *testing.B) {
	benchmarks := []struct {
		name string
		size int64
	}{
		{name: "10", size: 10},
		{name: "100", size: 100},
		{name: "1000", size: 1000},
		{name: "10000", size: 10000},
		{name: "100000", size: 100000},
	}
	for _, bm := range benchmarks {
		b.Run(bm.name+"-double", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(float64(benchmarkCompression), benchmarkDecayValue, benchmarkDecayEvery)
			td2 := NewWithDecay(float64(benchmarkCompression), benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1)
				td2.Add(data[i], 1)
				if int64(i)%bm.size == 0 {
					td2.Clear()
				}
			}
		})
		b.Run(bm.name+"-merge", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(float64(benchmarkCompression), benchmarkDecayValue, benchmarkDecayEvery)
			td2 := NewWithDecay(float64(benchmarkCompression), benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td2.Add(data[i], 1)
				if int64(i)%bm.size == 0 && i != 0 {
					td.AddCentroidList(td2.Centroids())
					td2.Clear()
				}
			}
			if td2.Centroids().Len() > 0 {
				td.AddCentroidList(td2.Centroids())
			}
		})
		b.Run(bm.name+"-regular", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(float64(benchmarkCompression), benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1)
			}
		})
	}
}

func getData(b *testing.B) []float64 {
	data := make([]float64, b.N)
	for i := 0; i < b.N; i++ {
		data[i] = getVal()
	}
	b.ResetTimer()
	return data
}
