if(cux.value===""){
    cux.value= "603";
    
}
else {
    if(parseInt(cux.value)>2383 || parseInt(cux.value)<0){
    cuxerror.classList.remove('hidden');
    return false;
}}
if(select.value==='default'){
    select.classList.remove('hidden');
    return false;
}
if(select2.value==='default'){
    select2error.classList.remove('hidden');
    return false;
}