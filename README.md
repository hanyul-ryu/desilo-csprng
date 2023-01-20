# Desilo CSPRNG

## Install dependency packages

```powershell
pip install torch torchvision torchaudio tqdm torch torchvision torchaudio numpy pandas mpmath numpy matplotlib
```



## Install Desilo-csprng

To pre-compile the cuda extention, install the our python packages using `python setup.py install` in advance.

1. randround
    ```powershell
    cd randround
    python setup.py install
    ```
    
    
2. randint
    ```powershell
    cd randint
    python setup.py install
    ```
    
    
3. discrete_gaussian
    ```powershell
    cd discrete_gaussian
    python setup.py install
    ```
    
    
4. chacha20
    ```powershell
    cd chacha20
    python setup.py install
    ```
    
    





### example codes

### Init

```python
import torch
import numpy as np
from rng import csprng



rng = csprng(
    N=2**16, C=19, 
    sigma=3.2, devices=[0, 1]
)
```



### Discrete Gaussian

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from rng import csprng

rng = csprng(
    N=2**16, C=19, 
    sigma=3.2, devices=[0, 1]
)

dg = rng.discrete_gaussian_copied()



device_id = 0
series = dg[device_id].squeeze().cpu().numpy()

plt.title("Discrete Gaussian Distribution")
plt.hist(
    series, 
    bins=np.arange(series.min()-0.5, series.max()+0.5)
)
plt.grid()
plt.show()
print(series.min(), series.mean(), series.max())
```

![discrete gaussian](/Users/hanyul/Dropbox/00. Desilo/WORK/lPrimCKKS/rng/images/discrete_gaussian.png)



### Random int

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from rng import csprng

rng = csprng(
    N=2**16, C=19, 
    sigma=3.2, devices=[0, 1]
)

q = [list(range(3)), list(range(5))]
urand = rng.randint(amax=q, shift=0)

```



### Random int (Copied singloe row)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from rng import csprng

rng = csprng(
    N=2**16, C=19, 
    sigma=3.2, devices=[0, 1]
)

sk = rng.randint_copied(3, -1) # gen secKey

device_id = 0
series = sk[0].squeeze().cpu().numpy()
```





### Random bytes

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from rng import csprng

rng = csprng(
    N=2**16, C=19, 
    sigma=3.2, devices=[0, 1]
)

rb = rng.randbytes(C=19, L=None)
```

