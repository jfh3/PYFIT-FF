#ifndef READINPUT_H
#define READINPUT_H

int readinput(const char *progname);

int ReadDatabase (const char *file);

void CreateNeighborList ();

int ReadBOPParam(const char *file);

void CreateNeighborList(Struc_Data *&data, const int nset);

int ReadData(const char *file, Struc_Data *&data, int &n, int &m);

void read_Modified_Gis(char *fname);
void read_Modified_Gis_cpus(char *fname);

#endif // READINPUT_H
