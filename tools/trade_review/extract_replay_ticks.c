/* Extract selected positive FO LTPs from the receiver's read-only UDP journals. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
typedef int (__cdecl *decoder)(const unsigned char *,size_t,unsigned char *,size_t *,void *);
static unsigned u16(const unsigned char *p){return ((unsigned)p[0]<<8)|p[1];}
static uint32_t u32(const unsigned char *p){return ((uint32_t)u16(p)<<16)|u16(p+2);}
static unsigned char selected[1<<20];
static int packet(const unsigned char *p,size_t n,decoder unzip){
    unsigned char decompressed[65536];
    if(n<4)return 0;
    unsigned count=u16(p+2);size_t pos=4;
    if(count>200)return 0;
    for(unsigned i=0;i<count;i++){
        if(pos+2>n)return 0;
        size_t len=u16(p+pos);pos+=2;
        const unsigned char *msg=p+pos;size_t size;
        if(len){
            if(pos+len>n)return 0;
            size=sizeof decompressed;
            if(unzip(msg,len,decompressed,&size,NULL)!=0)return 0;
            pos+=len;msg=decompressed;
        }else{
            if(pos+48>n)return 0;
            size=8+u16(msg+46);
            if(size<48 || pos+size>n)return 0;
            pos+=size;
        }
        if(size<50)continue;
        msg+=8;size-=8;
        unsigned code=u16(msg+10);
        if(code!=7208 && code!=17208)continue;
        unsigned stride=code==7208?214:258, records=u16(msg+40);
        if(records>2 || 42+records*stride>size)continue;
        for(unsigned j=0;j<records;j++){
            const unsigned char *rec=msg+42+j*stride;
            uint32_t token=u32(rec),price=u32(rec+(code==7208?12:16));
            if(token<sizeof selected && selected[token] && price && price<INT32_MAX){
                uint32_t tick[]={u32(msg+4),token,price};
                if(fwrite(tick,sizeof tick,1,stdout)!=1)return -1;
            }
        }
    }
    return 0;
}
int main(int argc,char **argv){
    if(argc<4)return 1;
    HMODULE lib=LoadLibraryA(argv[1]);
    if(!lib)return 2;
    decoder unzip=(decoder)GetProcAddress(lib,"lzo1z_decompress_safe");
    if(!unzip)return 3;
    for(char *p=strtok(argv[2],",");p;p=strtok(NULL,",")){
        long id=strtol(p,NULL,10);if(id<0 || (size_t)id>=sizeof selected)return 4;selected[id]=1;
    }
    _setmode(_fileno(stdout),_O_BINARY);
    unsigned char header[21],data[65536];
    for(int i=3;i<argc;i++){
        FILE *f=fopen(argv[i],"rb");if(!f)return 5;
        setvbuf(f,NULL,_IOFBF,1<<20);
        while(fread(header,1,sizeof header,f)==sizeof header){
            size_t len=u32(header+17);if(len>65535)return 6;
            if(fread(data,1,len,f)!=len)break;
            if(header[0]=='F' && packet(data,len,unzip)<0)return 7;
        }
        if(ferror(f))return 8;
        fclose(f);
    }
    return fflush(stdout)!=0;
}
