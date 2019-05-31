#ifndef CRYSTAL_STRUC_H
#define CRYSTAL_STRUC_H

void set_struc(const char *struc, const double scale, int &nb);

void fcc(const double scale, int &nb);

void diam(const double scale, int &nb);

void wurtzite(const double scale, int &nb);

void sc(const double scale, int &nb);

void bcc(const double scale, int &nb);

void betatin(const double scale, int &nb);

void cP46(const double scale, int &nb);

void ST12(const double scale, int &nb);

void BC8(const double scale, int &nb);

bool set_vac_supercell(const char *struc, const double scale, int &nb);

bool set_Td_int(const char *struc, const double scale, int &nb);

bool set_HEX_int(const char *struc, const double scale, int &nb);

bool set_B_int(const char *struc, const double scale, int &nb);

void set_dumbbell110_int(const char *struc, const double scale, int &nb);

void A15(const double scale, int &nb);

void hex(const double scale, int &nb);

void hcp(const double scale, int &nb);

void dimer(const double scale, int &nb);

void trimerD3h (const double scale, int &nb);

void trimerC2v (const double scale, int &nb);

void trimerDih (const double scale, int &nb);

void tetramerDih (const double scale, int &nb);

void tetramerD4h (const double scale, int &nb);

void tetramerTd (const double scale, int &nb);

void pentamerD5h (const double scale, int &nb);

void graphitic(const double scale, int &nb);

bool set_Octa_int(const char *struc, const double scale, int &nb);

void set_dumbbell100_int(const char *struc, const double scale, int &nb);

bool set_dumbbell_int(const char *struc, const char *type, const double scale, int &nb);

#endif // CRYSTAL_STRUC_H
