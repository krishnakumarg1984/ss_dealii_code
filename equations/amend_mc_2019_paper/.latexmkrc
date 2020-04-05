add_cus_dep( 'sage', 'sout', 0, 'makesout' );
$hash_calc_ignore_pattern{'sage'} = '^( _st_.goboom|print .SageT)';
sub makesout {
    system( "sage \"$_[0].sage\"" );
}
