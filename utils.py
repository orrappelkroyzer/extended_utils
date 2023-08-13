
def float2str(x, n_digits=0):
    x = str(_round(x, n_digits))
    return ".".join([format(int(x.split(".")[0]), ',d')] +  x.split(".")[1:])

def _round(x, n_digits=2):
    x = round(x, n_digits)
    if x == int(x):
        x = int(x)
    return x