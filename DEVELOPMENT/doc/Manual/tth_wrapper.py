#!/usr/bin/python

import sys
import re
import os
import subprocess as sp
import shutil

prestring=r'''\documentclass{book}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,xcolor,colortbl}

\usepackage{pxfonts}
\renewcommand*{\sfdefault}{lmss}

\usepackage[active,tightpage]{preview}
\PreviewEnvironment{equation}
\PreviewEnvironment{multline}
\PreviewEnvironment{align}
\setlength\PreviewBorder{100pt}
\setlength{\textwidth}{400pt}

\newcommand{\E}{\ensuremath{\mathrm{e}}}
\newcommand{\I}{\ensuremath{\mathrm{i}}}
\newcommand{\D}{\ensuremath{\mathrm{d}}}
\newcommand{\VEC}[1]{\ensuremath{\mathbf{#1}}}

\definecolor{RL}{HTML}{d65454}
\newcommand{\R}{\cellcolor{RL!50}}

\begin{document}

'''

poststring=r'''
\end{document}
'''

imagepath='html/'
temppath='tmp/'
pwd=os.getcwd()
mathenv=['equation','multline','align']
density_for_pdf={'sharc.pdf': 20}
tthpath='/user/mai/SHARC/REPOSITORY/DEVELOPMENT/doc/bin/tth'


# ================================================================================
# ================================================================================
# ================================================================================


# read input tex file
if len(sys.argv)==1:
  sys.stderr.write('Usage:\n./tth_wrapper.py <file.tex>\n')
  quit(0)
texname=sys.argv[1]
texf=open(texname)
tex=texf.readlines()
texf.close()


# make output directory
if not os.path.isdir(imagepath):
  os.mkdir(imagepath)
if not os.path.isdir(temppath):
  os.mkdir(temppath)


# open output tex file
# get filename
location=os.path.split(texname)[0]
if location!='':
  print 'Start me in the same directory as the .tex file!'
  exit(1)
basefilename=os.path.splitext(os.path.split(texname)[1])[0]

#tth_tex=os.path.join(temppath,basename+'.tex')
#tth_html=os.path.join(temppath,basename+'.html')
#tth_bib=os.path.join(temppath,basename)
#tth_html2=os.path.join(location,basename+'_final.html')
outf=open(os.path.join(temppath,basefilename+'_tth.tex'),'w')


# open equations tex file
#location=os.path.split(texname)[0]
eqbasename=basefilename+'_eq.tex'
eqf=open(os.path.join(temppath,basefilename+'_eq.tex'),'w')
eqf.write(prestring)


# parse the input file
iline=-1
eqc=-1
while True:
  iline+=1
  if iline==len(tex):
    break
  line=tex[iline]

  if line.strip()=='' or line.lstrip()[0]=='%':
    outf.write(line)
    continue

  # convert all \includegraphics
  # convert pdf to png, put all pngs to imagepath
  # change path in \includegraphics
  if '\includegraphics' in line:
    # get part before filename
    # get filename
    # get part after filename
    # assume that there is only ONE \includegraphics on each line
    preline=re.findall(r'^.*\includegraphics.*{',line)[0]
    postline=line[len(preline):-1]
    postline=re.findall(r'}.*$',postline)[0]
    filename=line[len(preline):-1][0:len(line)-len(preline)-len(postline)-1]
    # if pdf, generate png, else copy
    path=os.path.split(filename)[0]
    basename=os.path.splitext(os.path.split(filename)[1])[0]
    ext=os.path.splitext(os.path.split(filename)[1])[1]
    if ext=='.pdf':
      newfilename=os.path.join(imagepath,basename+'.png')
      if basename+ext in density_for_pdf:
        density=density_for_pdf[basename+ext]
      else:
        density=150
      shstring='convert -density %i -trim %s %s' % (density,filename,newfilename)
      print 'include PDF found: %s' % (filename)
      print '*** '+shstring+'\n'
      sp.call(shstring,shell=True)
    else:
      newfilename=os.path.join(imagepath,basename+ext)
      print 'include OTHER found: %s' % (filename)
      print '*** cp %s %s\n' % (filename,newfilename)
      shutil.copy(filename,newfilename)
    # write new line to output file
    newline=preline+newfilename+postline
    outf.write(newline)

  # convert equation environments to png via pdflatex and convert
  # copy them to imagepath
  # replace equation environment with includegraphics
  elif any([r'\begin{%s}' % (i) in line for i in mathenv]):
    print '%s found' % (line.strip())
    iline-=1
    while True:
      iline+=1
      line=tex[iline]
      eqf.write(line)
      if any([r'\end{%s}' % (i) in line for i in mathenv]):
        break
    # get filename
    eqc+=1
    newfilename=os.path.join(imagepath,'equation-%i.png' % (eqc))
    newline='\n\includegraphics{%s}\n' % (newfilename)
    outf.write(newline)

  # put chapters into the equation file for correct numbering
  elif r'\chapter' in line:
    outf.write(line)
    eqf.write(line)
    print 'New chapter\n'
  else:
    outf.write(line)

# typeset and convert all equations
outf.close()
eqf.write(poststring)
eqf.close()

os.chdir(temppath)
shline='pdflatex -interaction=nonstopmode %s > pdflatex.log 2> pdflatex.err' % (eqbasename)
print 'Running pdflatex'
print '*** %s\n' % (shline)
sp.call(shline,shell=True)
os.chdir(pwd)


pdffilename=os.path.join(temppath,basefilename+'_eq.pdf')
targetfilename=os.path.join(imagepath,'equation.png')
shline='convert -density 150 -trim %s %s' % (pdffilename,targetfilename)
print 'Running convert on all equations'
print 'This may take some time...'
print '*** %s\n' % (shline)
sp.call(shline,shell=True)


# run latex so that the aux, toc and bbl files are there
if any([not os.path.isfile(basefilename+'_tth.%s' % (i)) for i in ['aux','bbl','toc'] ]):
  shline='pdflatex -interaction=nonstopmode %s >> %s/pdflatex.log 2>> %s/pdflatex.err' % (basefilename+'.tex',temppath,temppath)
  bbline='bibtex %s >> %s/pdflatex.log 2>> %s/pdflatex.err' % (basefilename,temppath,temppath)

  print 'Running pdflatex and bibtex'
  print '*** %s\n' % (shline)
  sp.call(shline,shell=True)
  print '*** %s\n' % (bbline)
  sp.call(bbline,shell=True)
  print '*** %s\n' % (shline)
  sp.call(shline,shell=True)
  print '*** %s\n' % (shline)
  sp.call(shline,shell=True)



os.chdir(temppath)
if not os.path.isfile(basefilename+'_tth.aux'):
  print 'ln -s %s %s\n' % (pwd+'/'+basefilename+'.aux',basefilename+'_tth.aux')
  os.symlink(pwd+'/'+basefilename+'.aux',basefilename+'_tth.aux')
if not os.path.isfile(basefilename+'_tth.bbl'):
  print 'ln -s %s %s\n' % (pwd+'/'+basefilename+'.aux',basefilename+'_tth.bbl')
  os.symlink(pwd+'/'+basefilename+'.bbl',basefilename+'_tth.bbl')
if not os.path.isfile(basefilename+'_tth.toc'):
  print 'ln -s %s %s\n' % (pwd+'/'+basefilename+'.aux',basefilename+'_tth.toc')
  os.symlink(pwd+'/'+basefilename+'.toc',basefilename+'_tth.toc')


shline='%s %s > tth.log 2> tth.err' % (tthpath,basefilename+'_tth.tex')
print 'Running tth'
print '*** %s\n' % (shline)
sp.call(shline,shell=True)


# post processing starts here
htmlf=open(basefilename+'_tth.html')
html=htmlf.readlines()
htmlf.close()

print '*'*40
print 'Postprocessing of %s' % (temppath+'/'+basefilename+'_tth.html')
print '*'*40

os.chdir(pwd)
htmls=''


imagepath2='http://sharc-md.org/wp-content/uploads/2014/10'


write=False
iline=-1
while True:
  iline+=1
  if iline==len(html):
    break
  line=html[iline]

  if r'<title>' in line:
    # start after title
    write=True
  elif r'<small>File translated from' in line:
    # do not write from here anymore
    write=False

  if not write:
    continue

  if line.strip()=='':
    # skip empty lines
    continue

  if re.search(r'<a href=".*">Figure</a>', line):
    # substitute href with img src
    print 'Detected href to image'
    newline=re.sub(r'<a href="html/(.*)">Figure<\/a>',r'<img src="%s/\1">' % (imagepath2),line)
  elif re.search(r'<br />', line):
    # delete <br />
    print 'deleted <br />'
    newline=re.sub(r'<br />','',line)
  else:
    newline=line

  if newline[0]=='>' or r'<title>' in line:
    # no newlines before a >
    print 'stripping newline'
    htmls+=newline.rstrip('\n')
  else:
    htmls+='\n'+newline.rstrip('\n')

print '*'*40
print 'Find final output at %s' % (basefilename+'.html')
print '*'*40

htmlf=open(basefilename+'.html','w')
htmlf.write(htmls)
htmlf.close()

