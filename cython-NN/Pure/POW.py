import hashlib as hs


def POW(data, target):
    header = 1
    while True:
        
        for nonce in range(2**32):
            t = (data+nonce.to_bytes(32,"little")+header.to_bytes(64,"little"))
            if int(hs.sha256(t).hexdigest(), 16)< target:

                return (header,nonce)
        header+=1