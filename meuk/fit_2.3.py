



# 1000
aa = [  (0.31 + 0.30)/2,
        (1.14 + 1.16)/2,
        (4.26 + 4.36)/2,
        (16.5 + 16.5)/2]

# 2000
bb = [  (0.61 + 0.59)/2,
        (2.27 + 2.24)/2,
        (8.22 + 8.17)/2,
        (32.8 + 32.5)/2]



for a,b in zip(aa,bb):
    beta = (b-a)/1000
    alpha = 2*a - b

    print(f"a= {alpha:.3f}, b={beta:.6f}")

