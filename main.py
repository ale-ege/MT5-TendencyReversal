import subprocess
import sys

arquivos = [
    'WEGE3.py', 'EMBR3.py', 'AZUL4.py', 'RADL3.py', 'ABEV3.py',
    'ASAI3.py', 'CRFB3.py', 'BRFS3.py', 'JBSS3.py', 'MRFG3.py',
    'ITUB4.py', 'BBDC4.py', 'BBAS3.py', 'PETR4.py', 'VALE3.py', 'RENT3.py'
]


processos = []

for arq in arquivos:
    p = subprocess.Popen([sys.executable, arq])
    processos.append(p)

for p in processos:
    p.wait()
