import jax
import jax.numpy as jnp
import numpy as np

def f(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array([
            - x*x*y*z
            + x*x*y
            + x*x*z
            - x*z*z
            - 2*x*x+x*z
            + z*z+x-z,

            - y*y*z*z
            + 2*y*y*z
            - y*z*z
            - x*y+x*z
            - y*y+z*z+y
            - z
    ])

def descend(v):
    for _  in range(20):
        fv, f1v = f(v), f1(v)
        print(v, fv, f1v)
        # v = v - f1v . (f1v.T . f1v^-1 ) . fv
        v -= jnp.dot(jnp.dot(f1v.T, jnp.linalg.inv(jnp.dot(f1v, f1v.T))), fv)

f1 = jax.jacfwd(f, holomorphic=True)

# TODO: sample from sphere
vr = np.random.normal(size=(3,2))
v = (vr[:,0] + 1j * vr[:,1])
descend(v)
print(v, f1(v))





