"""NSE multicast feed: packet unpacking, LZO decompression, message decoders, raw journal.

    lzo.py        Lzo1zDecoder over native/lzo2.dll (rebuild with native/build_lzo.ps1)
    udp.py        NseFoUdpReceiver: multicast socket and sequential-packet unpacking
    journal.py    FeedJournal: minute-segmented raw datagram journal for replay
    order_book.py BankNiftyOrderBook: latest depth per option token
    messages/     decoders for 7207/7216 indices, 7208/17208 market-by-price, 720x OI
"""
