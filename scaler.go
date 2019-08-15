package tdigest

import "math"

type scaler interface {
	/**
	 * Computes q as a function of k. This is often faster than finding k as a function of q for some scales.
	 *
	 * @param k          The index value to convert into q scale.
	 * @param normalizer The normalizer value which depends on compression and (possibly) number of points in the
	 *                   digest.
	 * @return The value of q that corresponds to k
	 */
	q(k, normalizer float64) float64
	/**
	 * Converts  a quantile to the k-scale. The normalizer value depends on compression and (possibly) number of points
	 * in the digest. #normalizer(double, double)
	 *
	 * @param q          The quantile
	 * @param normalizer The normalizer value which depends on compression and (possibly) number of points in the
	 *                   digest.
	 * @return The corresponding value of k
	 */
	k(q, normalizer float64) float64
	/**
	 * Computes the normalizer given compression and number of points.
	 */
	normalizer(compression, n float64) float64
}

/**
 * Generates cluster sizes proportional to sqrt(q*(1-q)). This gives constant relative accuracy if accuracy is
 * proportional to squared cluster size. It is expected that K_2 and K_3 will give better practical results.
 */
type K1 struct{}

var _ scaler = &K1{}

func (*K1) q(k, normalizer float64) float64 {
	return (math.Sin(k/normalizer) + 1) / 2
}

func (*K1) k(q, normalizer float64) float64 {
	return normalizer * math.Asin(2*q-1)
}

func (*K1) normalizer(compression, n float64) float64 {
	return compression / (2 * math.Pi)
}

/**
 * Generates cluster sizes proportional to sqrt(q*(1-q)) but avoids computation of asin in the critical path by
 * using an approximate version.
 */
type K1Fast struct{}

var _ scaler = &K1Fast{}

func (*K1Fast) q(k, normalizer float64) float64 {
	return (math.Sin(k/normalizer) + 1) / 2
}

func (*K1Fast) k(q, normalizer float64) float64 {
	return normalizer * fastAsin(2*q-1)
}

func (*K1Fast) normalizer(compression, n float64) float64 {
	return compression / (2 * math.Pi)
}

const splitPoint = 0.5

/**
 * K1Spliced generates cluster sizes proportional to sqrt(1-q) for q >= 1/2, and uniform cluster sizes for q < 1/2 by gluing
 * the graph of the K_1 function to its tangent line at q=1/2. Changing the split point is possible.
 */
type K1Spliced struct{}

var _ scaler = &K1Spliced{}

func (*K1Spliced) q(k, normalizer float64) float64 {
	if k <= normalizer*math.Asin(2*splitPoint-1) {
		ww := k/normalizer - math.Asin(2*splitPoint-1)
		return ww*math.Sqrt(splitPoint*(1-splitPoint)) + splitPoint
	} else {
		return (math.Sin(k/normalizer) + 1) / 2
	}
}

func (*K1Spliced) k(q, normalizer float64) float64 {
	if q <= splitPoint {
		return normalizer * (math.Asin(2*splitPoint-1) + (q-splitPoint)/(math.Sqrt(splitPoint*(1-splitPoint))))
	} else {
		return normalizer * math.Asin(2*q-1)
	}
}

func (*K1Spliced) normalizer(compression, n float64) float64 {
	return compression / (2 * math.Pi)
}

type K1SplicedFast struct{}

var _ scaler = &K1SplicedFast{}

func (*K1SplicedFast) q(k, normalizer float64) float64 {
	if k <= normalizer*fastAsin(2*splitPoint-1) {
		ww := k/normalizer - fastAsin(2*splitPoint-1)
		return ww*math.Sqrt(splitPoint*(1-splitPoint)) + splitPoint
	} else {
		return (math.Sin(k/normalizer) + 1) / 2
	}
}

func (*K1SplicedFast) k(q, normalizer float64) float64 {
	if q <= splitPoint {
		return normalizer * (fastAsin(2*splitPoint-1) + (q-splitPoint)/(math.Sqrt(splitPoint*(1-splitPoint))))
	} else {
		return normalizer * fastAsin(2*q-1)
	}
}

func (*K1SplicedFast) normalizer(compression, n float64) float64 {
	return compression / (2 * math.Pi)
}

/**
 * Generates cluster sizes proportional to q*(1-q). This makes tail error bounds tighter than for K_1. The use of a
 * normalizing function results in a strictly bounded number of clusters no matter how many samples.
 */
type K2 struct{}

var _ scaler = &K2{}

func (*K2) q(k, normalizer float64) float64 {
	w := math.Exp(k / normalizer)
	return w / (1 + w)
}

func (t *K2) k(q, normalizer float64) float64 {
	if q < 1e-15 {
		// this will return something more extreme than q = 1/n
		return 2 * t.k(1e-15, normalizer)
	} else if q > 1-1e-15 {
		// this will return something more extreme than q = (n-1)/n
		return 2 * t.k(1-1e-15, normalizer)
	} else {
		return math.Log(q/(1-q)) * normalizer
	}
}

func (t *K2) normalizer(compression, n float64) float64 {
	return compression / Z24(compression, n)
}

func Z24(compression, n float64) float64 {
	return 4*math.Log(n/compression) + 24
}

/**
 * Generates cluster sizes proportional to 1-q for q >= 1/2, and uniform cluster sizes for q < 1/2 by gluing
 * the graph of the K_2 function to its tangent line at q=1/2. Changing the split point is possible.
 */
type K2Spliced struct{}

var _ scaler = &K2Spliced{}

func (*K2Spliced) q(k, normalizer float64) float64 {
	if k <= math.Log(splitPoint/(1-splitPoint))/normalizer {
		return (1-splitPoint)*(k/normalizer-math.Log(splitPoint/(1-splitPoint))) + splitPoint
	} else {
		w := math.Exp(k / normalizer)
		return w / (1 + w)
	}
}

func (t *K2Spliced) k(q, normalizer float64) float64 {
	if q <= splitPoint {
		return ((q - splitPoint) / splitPoint / (1 - splitPoint)) + math.Log(splitPoint/(1-splitPoint))*normalizer
	} else if q > 1-1e-15 {
		// this will return something more extreme than q = (n-1)/n
		return 2 * t.k(1-1e-15, normalizer)
	} else {
		return math.Log(q/(1-q)) * normalizer
	}
}

func (t *K2Spliced) normalizer(compression, n float64) float64 {
	return compression / Z24(compression, n)
}

/**
 * Generates cluster sizes proportional to min(q, 1-q). This makes tail error bounds tighter than for K_1 or K_2.
 * The use of a normalizing function results in a strictly bounded number of clusters no matter how many samples.
 */
type K3 struct{}

var _ scaler = &K3{}

func (t *K3) q(k, normalizer float64) float64 {
	if k <= 0 {
		return math.Exp(k/normalizer) / 2
	} else {
		return 1 - t.q(-k, normalizer)
	}
}

func (t *K3) k(q, normalizer float64) float64 {
	if q < 1e-15 {
		return 10 * t.k(1e-15, normalizer)
	} else if q > 1-1e-15 {
		return 10 * t.k(1-1e-15, normalizer)
	} else {
		if q <= 0.5 {
			return math.Log(2*q) / normalizer
		} else {
			return -t.k(1-q, normalizer)
		}
	}
}

func (t *K3) normalizer(compression, n float64) float64 {
	return compression / Z21(compression, n)
}

func Z21(compression, n float64) float64 {
	return 4*math.Log(n/compression) + 21
}

/**
 * Generates cluster sizes proportional to 1-q for q >= 1/2, and uniform cluster sizes for q < 1/2 by gluing
 * the graph of the K_3 function to its tangent line at q=1/2.
 */
type K3Spliced struct{}

var _ scaler = &K3Spliced{}

func (t *K3Spliced) q(k, normalizer float64) float64 {
	if k <= 0 {
		return ((k / normalizer) + 1) / 2
	} else {
		return 1 - (math.Exp(-k/normalizer) / 2)
	}
}

func (t *K3Spliced) k(q, normalizer float64) float64 {
	if q <= 0.5 {
		return normalizer * (2*q - 1)
	} else if q > 1-1e-15 {
		return 10 * t.k(1-1e-15, normalizer)
	} else {
		return -normalizer * math.Log(2*(1-q))
	}
}

func (t *K3Spliced) normalizer(compression, n float64) float64 {
	return compression / Z21(compression, n)
}

/**
 * Generates cluster sizes proportional to 1-q for q >= 1/2, and uniform cluster sizes for q < 1/2 by gluing
 * the graph of the K_3 function to its tangent line at q=1/2.
 */
type KQuadratic struct{}

var _ scaler = &KQuadratic{}

func (t *KQuadratic) q(k, normalizer float64) float64 {
	return math.Sqrt(normalizer*(normalizer+3*k))/normalizer - 1
}

func (t *KQuadratic) k(q, normalizer float64) float64 {
	return normalizer * (q*q + 2*q) / 3
}

func (*KQuadratic) normalizer(compression, n float64) float64 {
	return compression / 2
}
