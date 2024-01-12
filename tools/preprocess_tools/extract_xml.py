#!/usr/bin/env python3

# Adapted from Apertium TMX tools
# http://wiki.apertium.org/wiki/Tools_for_TMX
#
# Extract parallel segments from tmx file using sax parser
# so we can stream very large files

from xml.sax import make_parser
from xml.sax.handler import ContentHandler
from optparse import OptionParser

import codecs
import sys
import re

version = "0.0.1"

class TMXHandler(ContentHandler):
        
    def __init__ (self):
        self.inTag = ''
        self.seg = ''

    def startElement(self, name, attrs): 
        self.inTag = name
        if name == 'seg': 
            self.seg = ''
                    
    def characters (self, c): 
        if self.inTag == 'seg': 
            self.seg += c

    def endElement(self, name): 
        if name == 'seg': 

            if(options.lineclean):
                curseg = re.sub('\s+', ' ', self.seg.strip().replace('\n', ' '))
            else:
                curseg = self.seg.strip()

            out.write("%s\n" % curseg)

                    
if __name__ == '__main__':

    # Parse command line
    cl = OptionParser(version="%prog " + version)

    cl.add_option("-b", "--base", dest="base", help="base output filename")
    cl.add_option("-l", "--lang", dest="lang", help="source language")
    cl.add_option("-c", "--clean", dest="lineclean", help="clean newlines, spaces", action="store_true")

    (options, args) = cl.parse_args()

    parser = make_parser()

    # open output files
    out = codecs.open("%s.%s" % (options.base, options.lang), 
                         "w", "utf-8")

    curHandler = TMXHandler()

    parser.setContentHandler(curHandler)

    parser.parse(sys.stdin)