#!/usr/bin/python

import sys
import re

prestring=r'''\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,xcolor,colortbl}

\usepackage[scaled=.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sfmath}

\usepackage[active,tightpage]{preview}
\PreviewEnvironment{equation}
\PreviewEnvironment{multline}
\PreviewEnvironment{align}
\setlength\PreviewBorder{100pt}
\setlength{\textwidth}{246pt}

\newcommand{\E}{\ensuremath{\mathrm{e}}}
\newcommand{\I}{\ensuremath{\mathrm{i}}}
\newcommand{\D}{\ensuremath{\mathrm{d}}}
\renewcommand{\vec}[1]{\ensuremath{\mathbf{#1}}}

\definecolor{RL}{HTML}{d65454}
\newcommand{\R}{\cellcolor{RL!50}}

\begin{document}

'''

poststring=r'''
\end{document}
'''

if len(sys.argv)==1:
  sys.stderr.write('Usage:\n./make_equations.py <file.tex>\n')
  quit(0)

texname=sys.argv[1]
tex=open(texname)

script=open('make_all.sh','w')

eqc=0

mathenv=['equation','multline','align']


while True:
  s=tex.readline()
  if r'\end{document}' in s:
    break
  if any([r'\begin{%s}' % (i) in s for i in mathenv]):
    eqc+=1
    eq=open('equation_%i.tex' % eqc, 'w')
    eq.write(prestring)
    eq.write(s)
    while True:
      try:
        s=tex.readline()
      except EOFError:
        exit(1)
      s=re.sub('\\label\{.*\}','',s)
      s=s.replace(r'\\',r'\nonumber\\')
      if any([r'\end{%s}' % (i) in s for i in mathenv]):
        eq.write(r'\nonumber')
      eq.write(s)
      if any([r'\end{%s}' % (i) in s for i in mathenv]):
        break
    eq.write(poststring)
    eq.close()
    script.write('pdflatex equation_%i.tex\npdfcrop equation_%i.pdf\nmv equation_%i-crop.pdf equation_%i.pdf\nconvert -density 200 equation_%i.pdf equation_%i.gif\n' % (eqc,eqc,eqc,eqc,eqc,eqc))

tex.close()
script.write('rm *.aux *.log *.pdf')
script.close()