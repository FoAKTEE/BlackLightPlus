"""orbit file: out q  x1(3) v1(3) a1(3)  x2(3) v2(3) chi2(3); three uniform times, constant state."""
import sys, struct
a = sys.argv; out = a[1]; q = float(a[2]); v = list(map(float, a[3:21]))
x1, v1, a1, x2, v2, chi2 = v[0:3], v[3:6], v[6:9], v[9:12], v[12:15], v[15:18]
with open(out, 'wb') as f:
    f.write(f"3 {q}\n".encode())
    for t in (0.0, 1000.0, 2000.0):
        row = [t, *x1, *x2, *a1, *chi2, *v1, *v2]
        f.write(struct.pack('<' + 'd' * len(row), *row))
print(out)
