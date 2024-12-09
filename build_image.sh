#!/bin/bash
buildah build -t registry.cjbal.com/tesseract:latest --layers
buildah push registry.cjbal.com/tesseract:latest