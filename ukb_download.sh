#!/bin/bash

#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>


#download the necessary helper utilities from the ukbiobank
#wget -nd biobank.ndph.ox.ac.uk/ukb/util/ukbunpack
#wget -nd biobank.ndph.ox.ac.uk/ukb/util/dconvert
#wget -nd biobank.ndph.ox.ac.uk/ukb/util/ukbconv
#wget -nd biobank.ndph.ox.ac.uk/ukb/util/ukbfetch
#wget -nd biobank.ndph.ox.ac.uk/ukb/util/ukbmd5
#wget -nd biobank.ndph.ox.ac.uk/ukb/ukb/utilx/encoding.dat

#the command that pulls encoding.dat should produce a file like the one below with suffix .enc
./ukbmd5 ukb<appNum>/<encUKBfile>.enc

#then we want to link our encoding with the key value that can be downloaded from an email sent to the PI
./ukbunpack ukb<appNum>/<encUKBfile>.enc <key/.ukbkey>

#there should now be a file with the suffic .enc_ukb
./ukbconv ukb<appNum>.enc_ukb txt #options here is replaced by the desired data type: csv,txt,r,sas,stata

#alternatively data level transfer option 2 is now possible u50041s
./ukbconv ukb<appNum>/ukb<appNum>.enc_ukb docs -i files.we.want.txt #where the txt file contains the field ids of each field we want, each on a new line
