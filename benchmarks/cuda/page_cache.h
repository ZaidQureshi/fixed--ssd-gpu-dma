#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#include "util.h"
#include <nvm_types.h>
#include "buffer.h"
#include "ctrl.h"
#include "settings.h"
#include <iostream>

/*

enum page_state {USE = 1, USE_DIRTY = ((1 << 63) | 1), VALID_DIRTY = (1 << 63), VALID = 0, INVALID = (UINT_MAX & 0x7fffffffffffffff), BUSY = ((UINT_MAX & 0x7fffffffffffffff)-1)};

template <typname T>
struct range_t {
    uint64_t index_start;
    uint64_t index_end;

    uint64_t blk_start;
    uint64_t blk_end;
    page_cache_t* cache;
    padded_struct* page_states;
    padded_struct* page_addresses;
    //padded_struct* page_vals;  //len = num of pages for data
    //

    __device__
    uint64_t find_slot(uint64_t global_address) {
        bool fail = true;
        uint64_t count = 0;
        do {
            uint64_t page = cache->page_ticket.fetch_add(1, simt::memory_order_acquire)  & (cache->n_pages_minus_1);
            uint64_t unlocked = UNLOCKED;
            bool lock = cache->page_take_lock[page].compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                uint64_t previous_address = cache->page_translation[page].exchange(global_address, simt::memory_order_acquire);
                uint64_t expected_state = VALID;
                uint64_t new_state = BUSY;
                bool pass = page_states[previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                switch(expected_state) {
                    case VALID:
                        page_state[previous_address].store(INVALID, simt::memory_order_release);
                        fail = false;
                        break;
                    case INVALID:
                        pass =  page_states[previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if (pass) {
                            page_state[previous_address].store(INVALID, simt::memory_order_release);
                            fail = false;
                        }
                        break;
                    case VALID_DIRTY:


                        if ((count > cache->n_pages)) {
                            pass =  page_states[previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                            if  (pass) {
                                //if ((cache->page_dirty_start[page].load(simt::memory_order_acquire) == cache->page_dirty_end[page].load(simt::memory_order_acquire))) {

                                    //writeback
                                    page_state[previous_address].store(INVALID, simt::memory_order_release);

                                    fail = false;
                                //}
                                //else {
                                //    page_states[previous_address].store(expected_state, simt::memory_order_release);
                                //}
                            }
                        }
                        break;
                    default:

                        break;

                }
                if (fail)
                    cache->page_translation[page].store(previous_address, simt::memory_order_release);
                cache->page_take_lock[page].store(UNLOCKED, simt::memory_order_release);

            }

            count++;


        } while(fail);
    }
    __device__
    T operator[](size_t i) const {
        uint64_t index = (i * sizeof(T)) >> (cache->page_size_log);
        uint64_t subindex = (i * sizeof(T)) & (cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE;
        bool fail = true;
        T ret;
        do {
            bool pass = page_states[index].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            switch (expected_state) {
                case VALID:
                    uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                    while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        __nanosleep(100);
                    ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                    page_states[index].fetch_sub(1, simt::memory_order_release);
                    fail = false;
                    break;
                case BUSY:
                    expected_state = VALID;
                    new_state = USE;
                    break;
                case INVALID:
                    pass = page_states[index].compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        //find slot
                        //fill in
                        uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                        while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                            __nanosleep(100);
                        ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                        page_states[index].store(VALID, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = INVALID;
                        new_state = BUSY;
                    }

                    
                    break;
                default:
                    new_state = expected_state + 1;
                    pass = page_states[index].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                        while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                            __nanosleep(100);
                        ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                        page_states[index].fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = VALID;
                        new_state = USE;
                    }
                    break;
            }

        } while (fail);
        return ret;
    }

    __device__
    void operator()(size_t i, T val) {
        uint64_t index = (i * sizeof(T)) >> (cache->page_size_log);
        uint64_t subindex = (i * sizeof(T)) & (cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE_DIRTY;
        bool fail = true;
        T ret;
        do {
            bool pass = page_states[index].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            switch (expected_state) {
                case VALID:
                    uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                    while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        __nanosleep(100);
                    ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex] = val;
                    page_states[index].fetch_sub(1, simt::memory_order_release);
                    fail = false;
                    break;
                case BUSY:
                    expected_state = VALID;
                    new_state = USE_DIRTY;
                    break;
                case INVALID:
                    pass = page_states[index].compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        //find slot
                        //fill in
                        uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                        while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                            __nanosleep(100);
                        ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex] = val;
                        page_states[index].store(VALID, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = INVALID;
                        new_state = BUSY;
                    }
                    break;
                default:
                    new_state = (expected_state + 1) | 0x8000000000000000;
                    pass = page_states[index].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                        while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                            __nanosleep(100);
                        ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex] = val;
                        page_states[index].fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = VALID;
                        new_state = USE_DIRTY;
                    }
                    break;
            }

        } while (fail);
    } 
};

*/

struct page_cache_meta {
    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
};

struct page_cache_t {
    uint8_t* base_addr;
    uint32_t page_size;
    uint32_t page_size_log;
    uint64_t n_pages;
    padded_struct* page_translation;         //len = num of pages in cache
    padded_struct* page_use_start;      //len = num of pages in cache
    padded_struct* page_use_end;      //len = num of pages in cache
    padded_struct page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t* prp_list;              //len = num of pages in cache if page_size > ctrl.page_size *2
    uint64_t    ctrl_page_size;


    //BufferPtr prp2_list_buf;
    bool prps;
    page_cache_meta* meta;


    

page_cache_t(const uint32_t ps, const uint64_t np, const Settings& settings, const Controller& ctrl)
    : page_size(ps), n_pages(np), ctrl_page_size(ctrl.ctrl->page_size) {
        meta = (page_cache_meta*)malloc(sizeof(page_cache_meta));
        page_ticket.val = 0;
        uint64_t cache_size = ps*np;
        meta->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);
        base_addr = (uint8_t*) meta->pages_dma.get()->vaddr;
        std::cout << "HEREN\n";
        if (ps <= meta->pages_dma.get()->page_size) {
            meta->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) meta->prp1_buf.get();
            std::cout << np << " " << sizeof(uint64_t) << std::endl;
            uint64_t* temp = (uint64_t*) malloc(np * sizeof(uint64_t));
            if (temp == NULL)
                std::cout << "NULL\n";
            uint64_t how_many_in_one = meta->pages_dma.get()->page_size/ps;
            for (size_t i = 0; i < meta->pages_dma.get()->n_ioaddrs; i++) {
                for (size_t j = 0; j < how_many_in_one; j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)meta->pages_dma.get()->ioaddrs[i]) + j*ps;
                }
            }
            cuda_err_chk(cudaMemcpy(prp1, temp, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            std::cout << "HERE1\n";
            //free(temp);
            std::cout << "HERE2\n";
            prps = false;
        }

        else if ((ps > meta->pages_dma.get()->page_size) && (ps <= (meta->pages_dma.get()->page_size * 2))) {
            meta->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) meta->prp1_buf.get();
            meta->prp2_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp2 = (uint64_t*) meta->prp2_buf.get();
            uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            for (size_t i = 0; i < meta->pages_dma.get()->n_ioaddrs; i+=2) {
                temp1[i] = ((uint64_t)meta->pages_dma.get()->ioaddrs[i]);
                temp2[i] = ((uint64_t)meta->pages_dma.get()->ioaddrs[i+1]);
            }
            cuda_err_chk(cudaMemcpy(prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            prps = true;
        }
        else {
            meta->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) meta->prp1_buf.get();
            uint32_t prp_list_size = meta->pages_dma.get()->page_size * np;
            meta->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);
            meta->prp2_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp2 = (uint64_t*) meta->prp2_buf.get();
            uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp3 = (uint64_t*) malloc(prp_list_size);
            const uint32_t uints_per_page = meta->pages_dma.get()->page_size / sizeof(uint64_t);
            uint32_t how_many_in_one = ps/meta->pages_dma.get()->page_size;
            for (size_t i = 0; i < meta->pages_dma.get()->n_ioaddrs; i+=how_many_in_one) {
                temp1[i] = ((uint64_t)meta->pages_dma.get()->ioaddrs[i]);
                temp2[i] = ((uint64_t)meta->prp_list_dma.get()->ioaddrs[i]);
                for (size_t j = 0; j < (how_many_in_one-1); j++) {

                    temp3[(i/how_many_in_one)*uints_per_page + j] = ((uint64_t)meta->pages_dma.get()->ioaddrs[i+1+j]);
                }
            }
            cuda_err_chk(cudaMemcpy(prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(meta->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            free(temp3);
            prps = true;
        }
        std::cout << "Finish Making Page Cache\n";

    }



};




#endif // __PAGE_CACHE_H__
