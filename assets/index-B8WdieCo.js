function Th(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const a=Object.getOwnPropertyDescriptor(r,i);a&&Object.defineProperty(e,i,a.get?a:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const a of i)if(a.type==="childList")for(const o of a.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(i){const a={};return i.integrity&&(a.integrity=i.integrity),i.referrerPolicy&&(a.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?a.credentials="include":i.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(i){if(i.ep)return;i.ep=!0;const a=n(i);fetch(i.href,a)}})();var za=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function ro(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var tp={exports:{}},io={},np={exports:{}},Z={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Ti=Symbol.for("react.element"),Lh=Symbol.for("react.portal"),Ah=Symbol.for("react.fragment"),Rh=Symbol.for("react.strict_mode"),Ih=Symbol.for("react.profiler"),jh=Symbol.for("react.provider"),Oh=Symbol.for("react.context"),Mh=Symbol.for("react.forward_ref"),Dh=Symbol.for("react.suspense"),Fh=Symbol.for("react.memo"),Wh=Symbol.for("react.lazy"),Wu=Symbol.iterator;function Bh(e){return e===null||typeof e!="object"?null:(e=Wu&&e[Wu]||e["@@iterator"],typeof e=="function"?e:null)}var rp={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},ip=Object.assign,ap={};function Cr(e,t,n){this.props=e,this.context=t,this.refs=ap,this.updater=n||rp}Cr.prototype.isReactComponent={};Cr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};Cr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function op(){}op.prototype=Cr.prototype;function El(e,t,n){this.props=e,this.context=t,this.refs=ap,this.updater=n||rp}var Pl=El.prototype=new op;Pl.constructor=El;ip(Pl,Cr.prototype);Pl.isPureReactComponent=!0;var Bu=Array.isArray,sp=Object.prototype.hasOwnProperty,Nl={current:null},lp={key:!0,ref:!0,__self:!0,__source:!0};function up(e,t,n){var r,i={},a=null,o=null;if(t!=null)for(r in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(a=""+t.key),t)sp.call(t,r)&&!lp.hasOwnProperty(r)&&(i[r]=t[r]);var s=arguments.length-2;if(s===1)i.children=n;else if(1<s){for(var l=Array(s),u=0;u<s;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in s=e.defaultProps,s)i[r]===void 0&&(i[r]=s[r]);return{$$typeof:Ti,type:e,key:a,ref:o,props:i,_owner:Nl.current}}function qh(e,t){return{$$typeof:Ti,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function Tl(e){return typeof e=="object"&&e!==null&&e.$$typeof===Ti}function Uh(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var qu=/\/+/g;function No(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Uh(""+e.key):t.toString(36)}function da(e,t,n,r,i){var a=typeof e;(a==="undefined"||a==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(a){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case Ti:case Lh:o=!0}}if(o)return o=e,i=i(o),e=r===""?"."+No(o,0):r,Bu(i)?(n="",e!=null&&(n=e.replace(qu,"$&/")+"/"),da(i,t,n,"",function(u){return u})):i!=null&&(Tl(i)&&(i=qh(i,n+(!i.key||o&&o.key===i.key?"":(""+i.key).replace(qu,"$&/")+"/")+e)),t.push(i)),1;if(o=0,r=r===""?".":r+":",Bu(e))for(var s=0;s<e.length;s++){a=e[s];var l=r+No(a,s);o+=da(a,t,n,l,i)}else if(l=Bh(e),typeof l=="function")for(e=l.call(e),s=0;!(a=e.next()).done;)a=a.value,l=r+No(a,s++),o+=da(a,t,n,l,i);else if(a==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function Bi(e,t,n){if(e==null)return e;var r=[],i=0;return da(e,r,"","",function(a){return t.call(n,a,i++)}),r}function Vh(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var qe={current:null},pa={transition:null},Hh={ReactCurrentDispatcher:qe,ReactCurrentBatchConfig:pa,ReactCurrentOwner:Nl};function cp(){throw Error("act(...) is not supported in production builds of React.")}Z.Children={map:Bi,forEach:function(e,t,n){Bi(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return Bi(e,function(){t++}),t},toArray:function(e){return Bi(e,function(t){return t})||[]},only:function(e){if(!Tl(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};Z.Component=Cr;Z.Fragment=Ah;Z.Profiler=Ih;Z.PureComponent=El;Z.StrictMode=Rh;Z.Suspense=Dh;Z.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Hh;Z.act=cp;Z.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=ip({},e.props),i=e.key,a=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(a=t.ref,o=Nl.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var s=e.type.defaultProps;for(l in t)sp.call(t,l)&&!lp.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&s!==void 0?s[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){s=Array(l);for(var u=0;u<l;u++)s[u]=arguments[u+2];r.children=s}return{$$typeof:Ti,type:e.type,key:i,ref:a,props:r,_owner:o}};Z.createContext=function(e){return e={$$typeof:Oh,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:jh,_context:e},e.Consumer=e};Z.createElement=up;Z.createFactory=function(e){var t=up.bind(null,e);return t.type=e,t};Z.createRef=function(){return{current:null}};Z.forwardRef=function(e){return{$$typeof:Mh,render:e}};Z.isValidElement=Tl;Z.lazy=function(e){return{$$typeof:Wh,_payload:{_status:-1,_result:e},_init:Vh}};Z.memo=function(e,t){return{$$typeof:Fh,type:e,compare:t===void 0?null:t}};Z.startTransition=function(e){var t=pa.transition;pa.transition={};try{e()}finally{pa.transition=t}};Z.unstable_act=cp;Z.useCallback=function(e,t){return qe.current.useCallback(e,t)};Z.useContext=function(e){return qe.current.useContext(e)};Z.useDebugValue=function(){};Z.useDeferredValue=function(e){return qe.current.useDeferredValue(e)};Z.useEffect=function(e,t){return qe.current.useEffect(e,t)};Z.useId=function(){return qe.current.useId()};Z.useImperativeHandle=function(e,t,n){return qe.current.useImperativeHandle(e,t,n)};Z.useInsertionEffect=function(e,t){return qe.current.useInsertionEffect(e,t)};Z.useLayoutEffect=function(e,t){return qe.current.useLayoutEffect(e,t)};Z.useMemo=function(e,t){return qe.current.useMemo(e,t)};Z.useReducer=function(e,t,n){return qe.current.useReducer(e,t,n)};Z.useRef=function(e){return qe.current.useRef(e)};Z.useState=function(e){return qe.current.useState(e)};Z.useSyncExternalStore=function(e,t,n){return qe.current.useSyncExternalStore(e,t,n)};Z.useTransition=function(){return qe.current.useTransition()};Z.version="18.3.1";np.exports=Z;var C=np.exports;const jt=ro(C),$h=Th({__proto__:null,default:jt},[C]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Qh=C,Xh=Symbol.for("react.element"),Kh=Symbol.for("react.fragment"),Yh=Object.prototype.hasOwnProperty,Gh=Qh.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Zh={key:!0,ref:!0,__self:!0,__source:!0};function dp(e,t,n){var r,i={},a=null,o=null;n!==void 0&&(a=""+n),t.key!==void 0&&(a=""+t.key),t.ref!==void 0&&(o=t.ref);for(r in t)Yh.call(t,r)&&!Zh.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Xh,type:e,key:a,ref:o,props:i,_owner:Gh.current}}io.Fragment=Kh;io.jsx=dp;io.jsxs=dp;tp.exports=io;var x=tp.exports,_s={},pp={exports:{}},ot={},fp={exports:{}},mp={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(I,V){var y=I.length;I.push(V);e:for(;0<y;){var B=y-1>>>1,$=I[B];if(0<i($,V))I[B]=V,I[y]=$,y=B;else break e}}function n(I){return I.length===0?null:I[0]}function r(I){if(I.length===0)return null;var V=I[0],y=I.pop();if(y!==V){I[0]=y;e:for(var B=0,$=I.length,v=$>>>1;B<v;){var te=2*(B+1)-1,we=I[te],se=te+1,lt=I[se];if(0>i(we,y))se<$&&0>i(lt,we)?(I[B]=lt,I[se]=y,B=se):(I[B]=we,I[te]=y,B=te);else if(se<$&&0>i(lt,y))I[B]=lt,I[se]=y,B=se;else break e}}return V}function i(I,V){var y=I.sortIndex-V.sortIndex;return y!==0?y:I.id-V.id}if(typeof performance=="object"&&typeof performance.now=="function"){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var l=[],u=[],c=1,d=null,p=3,f=!1,w=!1,_=!1,S=typeof setTimeout=="function"?setTimeout:null,h=typeof clearTimeout=="function"?clearTimeout:null,m=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function g(I){for(var V=n(u);V!==null;){if(V.callback===null)r(u);else if(V.startTime<=I)r(u),V.sortIndex=V.expirationTime,t(l,V);else break;V=n(u)}}function b(I){if(_=!1,g(I),!w)if(n(l)!==null)w=!0,Q(z);else{var V=n(u);V!==null&&ee(b,V.startTime-I)}}function z(I,V){w=!1,_&&(_=!1,h(N),N=-1),f=!0;var y=p;try{for(g(V),d=n(l);d!==null&&(!(d.expirationTime>V)||I&&!F());){var B=d.callback;if(typeof B=="function"){d.callback=null,p=d.priorityLevel;var $=B(d.expirationTime<=V);V=e.unstable_now(),typeof $=="function"?d.callback=$:d===n(l)&&r(l),g(V)}else r(l);d=n(l)}if(d!==null)var v=!0;else{var te=n(u);te!==null&&ee(b,te.startTime-V),v=!1}return v}finally{d=null,p=y,f=!1}}var k=!1,E=null,N=-1,O=5,L=-1;function F(){return!(e.unstable_now()-L<O)}function U(){if(E!==null){var I=e.unstable_now();L=I;var V=!0;try{V=E(!0,I)}finally{V?Y():(k=!1,E=null)}}else k=!1}var Y;if(typeof m=="function")Y=function(){m(U)};else if(typeof MessageChannel<"u"){var re=new MessageChannel,j=re.port2;re.port1.onmessage=U,Y=function(){j.postMessage(null)}}else Y=function(){S(U,0)};function Q(I){E=I,k||(k=!0,Y())}function ee(I,V){N=S(function(){I(e.unstable_now())},V)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(I){I.callback=null},e.unstable_continueExecution=function(){w||f||(w=!0,Q(z))},e.unstable_forceFrameRate=function(I){0>I||125<I?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):O=0<I?Math.floor(1e3/I):5},e.unstable_getCurrentPriorityLevel=function(){return p},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(I){switch(p){case 1:case 2:case 3:var V=3;break;default:V=p}var y=p;p=V;try{return I()}finally{p=y}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(I,V){switch(I){case 1:case 2:case 3:case 4:case 5:break;default:I=3}var y=p;p=I;try{return V()}finally{p=y}},e.unstable_scheduleCallback=function(I,V,y){var B=e.unstable_now();switch(typeof y=="object"&&y!==null?(y=y.delay,y=typeof y=="number"&&0<y?B+y:B):y=B,I){case 1:var $=-1;break;case 2:$=250;break;case 5:$=1073741823;break;case 4:$=1e4;break;default:$=5e3}return $=y+$,I={id:c++,callback:V,priorityLevel:I,startTime:y,expirationTime:$,sortIndex:-1},y>B?(I.sortIndex=y,t(u,I),n(l)===null&&I===n(u)&&(_?(h(N),N=-1):_=!0,ee(b,y-B))):(I.sortIndex=$,t(l,I),w||f||(w=!0,Q(z))),I},e.unstable_shouldYield=F,e.unstable_wrapCallback=function(I){var V=p;return function(){var y=p;p=V;try{return I.apply(this,arguments)}finally{p=y}}}})(mp);fp.exports=mp;var Jh=fp.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var eg=C,at=Jh;function A(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var hp=new Set,li={};function Un(e,t){gr(e,t),gr(e+"Capture",t)}function gr(e,t){for(li[e]=t,e=0;e<t.length;e++)hp.add(t[e])}var Qt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),vs=Object.prototype.hasOwnProperty,tg=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,Uu={},Vu={};function ng(e){return vs.call(Vu,e)?!0:vs.call(Uu,e)?!1:tg.test(e)?Vu[e]=!0:(Uu[e]=!0,!1)}function rg(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function ig(e,t,n,r){if(t===null||typeof t>"u"||rg(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Ue(e,t,n,r,i,a,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=a,this.removeEmptyString=o}var Ae={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Ae[e]=new Ue(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Ae[t]=new Ue(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Ae[e]=new Ue(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Ae[e]=new Ue(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Ae[e]=new Ue(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Ae[e]=new Ue(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Ae[e]=new Ue(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Ae[e]=new Ue(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Ae[e]=new Ue(e,5,!1,e.toLowerCase(),null,!1,!1)});var Ll=/[\-:]([a-z])/g;function Al(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(Ll,Al);Ae[t]=new Ue(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(Ll,Al);Ae[t]=new Ue(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(Ll,Al);Ae[t]=new Ue(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Ae[e]=new Ue(e,1,!1,e.toLowerCase(),null,!1,!1)});Ae.xlinkHref=new Ue("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Ae[e]=new Ue(e,1,!1,e.toLowerCase(),null,!0,!0)});function Rl(e,t,n,r){var i=Ae.hasOwnProperty(t)?Ae[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(ig(t,n,i,r)&&(n=null),r||i===null?ng(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var Gt=eg.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,qi=Symbol.for("react.element"),Kn=Symbol.for("react.portal"),Yn=Symbol.for("react.fragment"),Il=Symbol.for("react.strict_mode"),ws=Symbol.for("react.profiler"),gp=Symbol.for("react.provider"),yp=Symbol.for("react.context"),jl=Symbol.for("react.forward_ref"),ks=Symbol.for("react.suspense"),bs=Symbol.for("react.suspense_list"),Ol=Symbol.for("react.memo"),en=Symbol.for("react.lazy"),xp=Symbol.for("react.offscreen"),Hu=Symbol.iterator;function Rr(e){return e===null||typeof e!="object"?null:(e=Hu&&e[Hu]||e["@@iterator"],typeof e=="function"?e:null)}var ye=Object.assign,To;function Vr(e){if(To===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);To=t&&t[1]||""}return`
`+To+e}var Lo=!1;function Ao(e,t){if(!e||Lo)return"";Lo=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),a=r.stack.split(`
`),o=i.length-1,s=a.length-1;1<=o&&0<=s&&i[o]!==a[s];)s--;for(;1<=o&&0<=s;o--,s--)if(i[o]!==a[s]){if(o!==1||s!==1)do if(o--,s--,0>s||i[o]!==a[s]){var l=`
`+i[o].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=o&&0<=s);break}}}finally{Lo=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?Vr(e):""}function ag(e){switch(e.tag){case 5:return Vr(e.type);case 16:return Vr("Lazy");case 13:return Vr("Suspense");case 19:return Vr("SuspenseList");case 0:case 2:case 15:return e=Ao(e.type,!1),e;case 11:return e=Ao(e.type.render,!1),e;case 1:return e=Ao(e.type,!0),e;default:return""}}function Ss(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case Yn:return"Fragment";case Kn:return"Portal";case ws:return"Profiler";case Il:return"StrictMode";case ks:return"Suspense";case bs:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case yp:return(e.displayName||"Context")+".Consumer";case gp:return(e._context.displayName||"Context")+".Provider";case jl:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Ol:return t=e.displayName||null,t!==null?t:Ss(e.type)||"Memo";case en:t=e._payload,e=e._init;try{return Ss(e(t))}catch{}}return null}function og(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return Ss(t);case 8:return t===Il?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function _n(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function _p(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function sg(e){var t=_p(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,a=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(o){r=""+o,a.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(o){r=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function Ui(e){e._valueTracker||(e._valueTracker=sg(e))}function vp(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=_p(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function Ea(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function Cs(e,t){var n=t.checked;return ye({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function $u(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=_n(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function wp(e,t){t=t.checked,t!=null&&Rl(e,"checked",t,!1)}function zs(e,t){wp(e,t);var n=_n(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?Es(e,t.type,n):t.hasOwnProperty("defaultValue")&&Es(e,t.type,_n(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function Qu(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function Es(e,t,n){(t!=="number"||Ea(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var Hr=Array.isArray;function lr(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+_n(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function Ps(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(A(91));return ye({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function Xu(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(A(92));if(Hr(n)){if(1<n.length)throw Error(A(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:_n(n)}}function kp(e,t){var n=_n(t.value),r=_n(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function Ku(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function bp(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function Ns(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?bp(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var Vi,Sp=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(Vi=Vi||document.createElement("div"),Vi.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=Vi.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function ui(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var Kr={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},lg=["Webkit","ms","Moz","O"];Object.keys(Kr).forEach(function(e){lg.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),Kr[t]=Kr[e]})});function Cp(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||Kr.hasOwnProperty(e)&&Kr[e]?(""+t).trim():t+"px"}function zp(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=Cp(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var ug=ye({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Ts(e,t){if(t){if(ug[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(A(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(A(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(A(61))}if(t.style!=null&&typeof t.style!="object")throw Error(A(62))}}function Ls(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var As=null;function Ml(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Rs=null,ur=null,cr=null;function Yu(e){if(e=Ri(e)){if(typeof Rs!="function")throw Error(A(280));var t=e.stateNode;t&&(t=uo(t),Rs(e.stateNode,e.type,t))}}function Ep(e){ur?cr?cr.push(e):cr=[e]:ur=e}function Pp(){if(ur){var e=ur,t=cr;if(cr=ur=null,Yu(e),t)for(e=0;e<t.length;e++)Yu(t[e])}}function Np(e,t){return e(t)}function Tp(){}var Ro=!1;function Lp(e,t,n){if(Ro)return e(t,n);Ro=!0;try{return Np(e,t,n)}finally{Ro=!1,(ur!==null||cr!==null)&&(Tp(),Pp())}}function ci(e,t){var n=e.stateNode;if(n===null)return null;var r=uo(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(A(231,t,typeof n));return n}var Is=!1;if(Qt)try{var Ir={};Object.defineProperty(Ir,"passive",{get:function(){Is=!0}}),window.addEventListener("test",Ir,Ir),window.removeEventListener("test",Ir,Ir)}catch{Is=!1}function cg(e,t,n,r,i,a,o,s,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(c){this.onError(c)}}var Yr=!1,Pa=null,Na=!1,js=null,dg={onError:function(e){Yr=!0,Pa=e}};function pg(e,t,n,r,i,a,o,s,l){Yr=!1,Pa=null,cg.apply(dg,arguments)}function fg(e,t,n,r,i,a,o,s,l){if(pg.apply(this,arguments),Yr){if(Yr){var u=Pa;Yr=!1,Pa=null}else throw Error(A(198));Na||(Na=!0,js=u)}}function Vn(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function Ap(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function Gu(e){if(Vn(e)!==e)throw Error(A(188))}function mg(e){var t=e.alternate;if(!t){if(t=Vn(e),t===null)throw Error(A(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return Gu(i),e;if(a===r)return Gu(i),t;a=a.sibling}throw Error(A(188))}if(n.return!==r.return)n=i,r=a;else{for(var o=!1,s=i.child;s;){if(s===n){o=!0,n=i,r=a;break}if(s===r){o=!0,r=i,n=a;break}s=s.sibling}if(!o){for(s=a.child;s;){if(s===n){o=!0,n=a,r=i;break}if(s===r){o=!0,r=a,n=i;break}s=s.sibling}if(!o)throw Error(A(189))}}if(n.alternate!==r)throw Error(A(190))}if(n.tag!==3)throw Error(A(188));return n.stateNode.current===n?e:t}function Rp(e){return e=mg(e),e!==null?Ip(e):null}function Ip(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Ip(e);if(t!==null)return t;e=e.sibling}return null}var jp=at.unstable_scheduleCallback,Zu=at.unstable_cancelCallback,hg=at.unstable_shouldYield,gg=at.unstable_requestPaint,ke=at.unstable_now,yg=at.unstable_getCurrentPriorityLevel,Dl=at.unstable_ImmediatePriority,Op=at.unstable_UserBlockingPriority,Ta=at.unstable_NormalPriority,xg=at.unstable_LowPriority,Mp=at.unstable_IdlePriority,ao=null,Ot=null;function _g(e){if(Ot&&typeof Ot.onCommitFiberRoot=="function")try{Ot.onCommitFiberRoot(ao,e,void 0,(e.current.flags&128)===128)}catch{}}var zt=Math.clz32?Math.clz32:kg,vg=Math.log,wg=Math.LN2;function kg(e){return e>>>=0,e===0?32:31-(vg(e)/wg|0)|0}var Hi=64,$i=4194304;function $r(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function La(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,a=e.pingedLanes,o=n&268435455;if(o!==0){var s=o&~i;s!==0?r=$r(s):(a&=o,a!==0&&(r=$r(a)))}else o=n&~i,o!==0?r=$r(o):a!==0&&(r=$r(a));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,a=t&-t,i>=a||i===16&&(a&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-zt(t),i=1<<n,r|=e[n],t&=~i;return r}function bg(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Sg(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes;0<a;){var o=31-zt(a),s=1<<o,l=i[o];l===-1?(!(s&n)||s&r)&&(i[o]=bg(s,t)):l<=t&&(e.expiredLanes|=s),a&=~s}}function Os(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function Dp(){var e=Hi;return Hi<<=1,!(Hi&4194240)&&(Hi=64),e}function Io(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function Li(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-zt(t),e[t]=n}function Cg(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-zt(n),a=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~a}}function Fl(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-zt(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var oe=0;function Fp(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var Wp,Wl,Bp,qp,Up,Ms=!1,Qi=[],ln=null,un=null,cn=null,di=new Map,pi=new Map,nn=[],zg="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function Ju(e,t){switch(e){case"focusin":case"focusout":ln=null;break;case"dragenter":case"dragleave":un=null;break;case"mouseover":case"mouseout":cn=null;break;case"pointerover":case"pointerout":di.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":pi.delete(t.pointerId)}}function jr(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Ri(t),t!==null&&Wl(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Eg(e,t,n,r,i){switch(t){case"focusin":return ln=jr(ln,e,t,n,r,i),!0;case"dragenter":return un=jr(un,e,t,n,r,i),!0;case"mouseover":return cn=jr(cn,e,t,n,r,i),!0;case"pointerover":var a=i.pointerId;return di.set(a,jr(di.get(a)||null,e,t,n,r,i)),!0;case"gotpointercapture":return a=i.pointerId,pi.set(a,jr(pi.get(a)||null,e,t,n,r,i)),!0}return!1}function Vp(e){var t=Rn(e.target);if(t!==null){var n=Vn(t);if(n!==null){if(t=n.tag,t===13){if(t=Ap(n),t!==null){e.blockedOn=t,Up(e.priority,function(){Bp(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function fa(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=Ds(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);As=r,n.target.dispatchEvent(r),As=null}else return t=Ri(n),t!==null&&Wl(t),e.blockedOn=n,!1;t.shift()}return!0}function ec(e,t,n){fa(e)&&n.delete(t)}function Pg(){Ms=!1,ln!==null&&fa(ln)&&(ln=null),un!==null&&fa(un)&&(un=null),cn!==null&&fa(cn)&&(cn=null),di.forEach(ec),pi.forEach(ec)}function Or(e,t){e.blockedOn===t&&(e.blockedOn=null,Ms||(Ms=!0,at.unstable_scheduleCallback(at.unstable_NormalPriority,Pg)))}function fi(e){function t(i){return Or(i,e)}if(0<Qi.length){Or(Qi[0],e);for(var n=1;n<Qi.length;n++){var r=Qi[n];r.blockedOn===e&&(r.blockedOn=null)}}for(ln!==null&&Or(ln,e),un!==null&&Or(un,e),cn!==null&&Or(cn,e),di.forEach(t),pi.forEach(t),n=0;n<nn.length;n++)r=nn[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<nn.length&&(n=nn[0],n.blockedOn===null);)Vp(n),n.blockedOn===null&&nn.shift()}var dr=Gt.ReactCurrentBatchConfig,Aa=!0;function Ng(e,t,n,r){var i=oe,a=dr.transition;dr.transition=null;try{oe=1,Bl(e,t,n,r)}finally{oe=i,dr.transition=a}}function Tg(e,t,n,r){var i=oe,a=dr.transition;dr.transition=null;try{oe=4,Bl(e,t,n,r)}finally{oe=i,dr.transition=a}}function Bl(e,t,n,r){if(Aa){var i=Ds(e,t,n,r);if(i===null)Vo(e,t,r,Ra,n),Ju(e,r);else if(Eg(i,e,t,n,r))r.stopPropagation();else if(Ju(e,r),t&4&&-1<zg.indexOf(e)){for(;i!==null;){var a=Ri(i);if(a!==null&&Wp(a),a=Ds(e,t,n,r),a===null&&Vo(e,t,r,Ra,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Vo(e,t,r,null,n)}}var Ra=null;function Ds(e,t,n,r){if(Ra=null,e=Ml(r),e=Rn(e),e!==null)if(t=Vn(e),t===null)e=null;else if(n=t.tag,n===13){if(e=Ap(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return Ra=e,null}function Hp(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(yg()){case Dl:return 1;case Op:return 4;case Ta:case xg:return 16;case Mp:return 536870912;default:return 16}default:return 16}}var an=null,ql=null,ma=null;function $p(){if(ma)return ma;var e,t=ql,n=t.length,r,i="value"in an?an.value:an.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return ma=i.slice(e,1<r?1-r:void 0)}function ha(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Xi(){return!0}function tc(){return!1}function st(e){function t(n,r,i,a,o){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=a,this.target=o,this.currentTarget=null;for(var s in e)e.hasOwnProperty(s)&&(n=e[s],this[s]=n?n(a):a[s]);return this.isDefaultPrevented=(a.defaultPrevented!=null?a.defaultPrevented:a.returnValue===!1)?Xi:tc,this.isPropagationStopped=tc,this}return ye(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Xi)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Xi)},persist:function(){},isPersistent:Xi}),t}var zr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Ul=st(zr),Ai=ye({},zr,{view:0,detail:0}),Lg=st(Ai),jo,Oo,Mr,oo=ye({},Ai,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Vl,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==Mr&&(Mr&&e.type==="mousemove"?(jo=e.screenX-Mr.screenX,Oo=e.screenY-Mr.screenY):Oo=jo=0,Mr=e),jo)},movementY:function(e){return"movementY"in e?e.movementY:Oo}}),nc=st(oo),Ag=ye({},oo,{dataTransfer:0}),Rg=st(Ag),Ig=ye({},Ai,{relatedTarget:0}),Mo=st(Ig),jg=ye({},zr,{animationName:0,elapsedTime:0,pseudoElement:0}),Og=st(jg),Mg=ye({},zr,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),Dg=st(Mg),Fg=ye({},zr,{data:0}),rc=st(Fg),Wg={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},Bg={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},qg={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function Ug(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=qg[e])?!!t[e]:!1}function Vl(){return Ug}var Vg=ye({},Ai,{key:function(e){if(e.key){var t=Wg[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=ha(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?Bg[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Vl,charCode:function(e){return e.type==="keypress"?ha(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?ha(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),Hg=st(Vg),$g=ye({},oo,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),ic=st($g),Qg=ye({},Ai,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Vl}),Xg=st(Qg),Kg=ye({},zr,{propertyName:0,elapsedTime:0,pseudoElement:0}),Yg=st(Kg),Gg=ye({},oo,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Zg=st(Gg),Jg=[9,13,27,32],Hl=Qt&&"CompositionEvent"in window,Gr=null;Qt&&"documentMode"in document&&(Gr=document.documentMode);var ey=Qt&&"TextEvent"in window&&!Gr,Qp=Qt&&(!Hl||Gr&&8<Gr&&11>=Gr),ac=" ",oc=!1;function Xp(e,t){switch(e){case"keyup":return Jg.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Kp(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var Gn=!1;function ty(e,t){switch(e){case"compositionend":return Kp(t);case"keypress":return t.which!==32?null:(oc=!0,ac);case"textInput":return e=t.data,e===ac&&oc?null:e;default:return null}}function ny(e,t){if(Gn)return e==="compositionend"||!Hl&&Xp(e,t)?(e=$p(),ma=ql=an=null,Gn=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return Qp&&t.locale!=="ko"?null:t.data;default:return null}}var ry={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function sc(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!ry[e.type]:t==="textarea"}function Yp(e,t,n,r){Ep(r),t=Ia(t,"onChange"),0<t.length&&(n=new Ul("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var Zr=null,mi=null;function iy(e){lf(e,0)}function so(e){var t=er(e);if(vp(t))return e}function ay(e,t){if(e==="change")return t}var Gp=!1;if(Qt){var Do;if(Qt){var Fo="oninput"in document;if(!Fo){var lc=document.createElement("div");lc.setAttribute("oninput","return;"),Fo=typeof lc.oninput=="function"}Do=Fo}else Do=!1;Gp=Do&&(!document.documentMode||9<document.documentMode)}function uc(){Zr&&(Zr.detachEvent("onpropertychange",Zp),mi=Zr=null)}function Zp(e){if(e.propertyName==="value"&&so(mi)){var t=[];Yp(t,mi,e,Ml(e)),Lp(iy,t)}}function oy(e,t,n){e==="focusin"?(uc(),Zr=t,mi=n,Zr.attachEvent("onpropertychange",Zp)):e==="focusout"&&uc()}function sy(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return so(mi)}function ly(e,t){if(e==="click")return so(t)}function uy(e,t){if(e==="input"||e==="change")return so(t)}function cy(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var Pt=typeof Object.is=="function"?Object.is:cy;function hi(e,t){if(Pt(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!vs.call(t,i)||!Pt(e[i],t[i]))return!1}return!0}function cc(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function dc(e,t){var n=cc(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=cc(n)}}function Jp(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Jp(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function ef(){for(var e=window,t=Ea();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Ea(e.document)}return t}function $l(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function dy(e){var t=ef(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&Jp(n.ownerDocument.documentElement,n)){if(r!==null&&$l(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,a=Math.min(r.start,i);r=r.end===void 0?a:Math.min(r.end,i),!e.extend&&a>r&&(i=r,r=a,a=i),i=dc(n,a);var o=dc(n,r);i&&o&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),a>r?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var py=Qt&&"documentMode"in document&&11>=document.documentMode,Zn=null,Fs=null,Jr=null,Ws=!1;function pc(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Ws||Zn==null||Zn!==Ea(r)||(r=Zn,"selectionStart"in r&&$l(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Jr&&hi(Jr,r)||(Jr=r,r=Ia(Fs,"onSelect"),0<r.length&&(t=new Ul("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=Zn)))}function Ki(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var Jn={animationend:Ki("Animation","AnimationEnd"),animationiteration:Ki("Animation","AnimationIteration"),animationstart:Ki("Animation","AnimationStart"),transitionend:Ki("Transition","TransitionEnd")},Wo={},tf={};Qt&&(tf=document.createElement("div").style,"AnimationEvent"in window||(delete Jn.animationend.animation,delete Jn.animationiteration.animation,delete Jn.animationstart.animation),"TransitionEvent"in window||delete Jn.transitionend.transition);function lo(e){if(Wo[e])return Wo[e];if(!Jn[e])return e;var t=Jn[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in tf)return Wo[e]=t[n];return e}var nf=lo("animationend"),rf=lo("animationiteration"),af=lo("animationstart"),of=lo("transitionend"),sf=new Map,fc="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function wn(e,t){sf.set(e,t),Un(t,[e])}for(var Bo=0;Bo<fc.length;Bo++){var qo=fc[Bo],fy=qo.toLowerCase(),my=qo[0].toUpperCase()+qo.slice(1);wn(fy,"on"+my)}wn(nf,"onAnimationEnd");wn(rf,"onAnimationIteration");wn(af,"onAnimationStart");wn("dblclick","onDoubleClick");wn("focusin","onFocus");wn("focusout","onBlur");wn(of,"onTransitionEnd");gr("onMouseEnter",["mouseout","mouseover"]);gr("onMouseLeave",["mouseout","mouseover"]);gr("onPointerEnter",["pointerout","pointerover"]);gr("onPointerLeave",["pointerout","pointerover"]);Un("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Un("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Un("onBeforeInput",["compositionend","keypress","textInput","paste"]);Un("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Un("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Un("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Qr="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),hy=new Set("cancel close invalid load scroll toggle".split(" ").concat(Qr));function mc(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,fg(r,t,void 0,e),e.currentTarget=null}function lf(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],l=s.instance,u=s.currentTarget;if(s=s.listener,l!==a&&i.isPropagationStopped())break e;mc(i,s,u),a=l}else for(o=0;o<r.length;o++){if(s=r[o],l=s.instance,u=s.currentTarget,s=s.listener,l!==a&&i.isPropagationStopped())break e;mc(i,s,u),a=l}}}if(Na)throw e=js,Na=!1,js=null,e}function pe(e,t){var n=t[Hs];n===void 0&&(n=t[Hs]=new Set);var r=e+"__bubble";n.has(r)||(uf(t,e,2,!1),n.add(r))}function Uo(e,t,n){var r=0;t&&(r|=4),uf(n,e,r,t)}var Yi="_reactListening"+Math.random().toString(36).slice(2);function gi(e){if(!e[Yi]){e[Yi]=!0,hp.forEach(function(n){n!=="selectionchange"&&(hy.has(n)||Uo(n,!1,e),Uo(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Yi]||(t[Yi]=!0,Uo("selectionchange",!1,t))}}function uf(e,t,n,r){switch(Hp(t)){case 1:var i=Ng;break;case 4:i=Tg;break;default:i=Bl}n=i.bind(null,t,n,e),i=void 0,!Is||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function Vo(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i||s.nodeType===8&&s.parentNode===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&(l=o.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;o=o.return}for(;s!==null;){if(o=Rn(s),o===null)return;if(l=o.tag,l===5||l===6){r=a=o;continue e}s=s.parentNode}}r=r.return}Lp(function(){var u=a,c=Ml(n),d=[];e:{var p=sf.get(e);if(p!==void 0){var f=Ul,w=e;switch(e){case"keypress":if(ha(n)===0)break e;case"keydown":case"keyup":f=Hg;break;case"focusin":w="focus",f=Mo;break;case"focusout":w="blur",f=Mo;break;case"beforeblur":case"afterblur":f=Mo;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":f=nc;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":f=Rg;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":f=Xg;break;case nf:case rf:case af:f=Og;break;case of:f=Yg;break;case"scroll":f=Lg;break;case"wheel":f=Zg;break;case"copy":case"cut":case"paste":f=Dg;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":f=ic}var _=(t&4)!==0,S=!_&&e==="scroll",h=_?p!==null?p+"Capture":null:p;_=[];for(var m=u,g;m!==null;){g=m;var b=g.stateNode;if(g.tag===5&&b!==null&&(g=b,h!==null&&(b=ci(m,h),b!=null&&_.push(yi(m,b,g)))),S)break;m=m.return}0<_.length&&(p=new f(p,w,null,n,c),d.push({event:p,listeners:_}))}}if(!(t&7)){e:{if(p=e==="mouseover"||e==="pointerover",f=e==="mouseout"||e==="pointerout",p&&n!==As&&(w=n.relatedTarget||n.fromElement)&&(Rn(w)||w[Xt]))break e;if((f||p)&&(p=c.window===c?c:(p=c.ownerDocument)?p.defaultView||p.parentWindow:window,f?(w=n.relatedTarget||n.toElement,f=u,w=w?Rn(w):null,w!==null&&(S=Vn(w),w!==S||w.tag!==5&&w.tag!==6)&&(w=null)):(f=null,w=u),f!==w)){if(_=nc,b="onMouseLeave",h="onMouseEnter",m="mouse",(e==="pointerout"||e==="pointerover")&&(_=ic,b="onPointerLeave",h="onPointerEnter",m="pointer"),S=f==null?p:er(f),g=w==null?p:er(w),p=new _(b,m+"leave",f,n,c),p.target=S,p.relatedTarget=g,b=null,Rn(c)===u&&(_=new _(h,m+"enter",w,n,c),_.target=g,_.relatedTarget=S,b=_),S=b,f&&w)t:{for(_=f,h=w,m=0,g=_;g;g=Qn(g))m++;for(g=0,b=h;b;b=Qn(b))g++;for(;0<m-g;)_=Qn(_),m--;for(;0<g-m;)h=Qn(h),g--;for(;m--;){if(_===h||h!==null&&_===h.alternate)break t;_=Qn(_),h=Qn(h)}_=null}else _=null;f!==null&&hc(d,p,f,_,!1),w!==null&&S!==null&&hc(d,S,w,_,!0)}}e:{if(p=u?er(u):window,f=p.nodeName&&p.nodeName.toLowerCase(),f==="select"||f==="input"&&p.type==="file")var z=ay;else if(sc(p))if(Gp)z=uy;else{z=sy;var k=oy}else(f=p.nodeName)&&f.toLowerCase()==="input"&&(p.type==="checkbox"||p.type==="radio")&&(z=ly);if(z&&(z=z(e,u))){Yp(d,z,n,c);break e}k&&k(e,p,u),e==="focusout"&&(k=p._wrapperState)&&k.controlled&&p.type==="number"&&Es(p,"number",p.value)}switch(k=u?er(u):window,e){case"focusin":(sc(k)||k.contentEditable==="true")&&(Zn=k,Fs=u,Jr=null);break;case"focusout":Jr=Fs=Zn=null;break;case"mousedown":Ws=!0;break;case"contextmenu":case"mouseup":case"dragend":Ws=!1,pc(d,n,c);break;case"selectionchange":if(py)break;case"keydown":case"keyup":pc(d,n,c)}var E;if(Hl)e:{switch(e){case"compositionstart":var N="onCompositionStart";break e;case"compositionend":N="onCompositionEnd";break e;case"compositionupdate":N="onCompositionUpdate";break e}N=void 0}else Gn?Xp(e,n)&&(N="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(N="onCompositionStart");N&&(Qp&&n.locale!=="ko"&&(Gn||N!=="onCompositionStart"?N==="onCompositionEnd"&&Gn&&(E=$p()):(an=c,ql="value"in an?an.value:an.textContent,Gn=!0)),k=Ia(u,N),0<k.length&&(N=new rc(N,e,null,n,c),d.push({event:N,listeners:k}),E?N.data=E:(E=Kp(n),E!==null&&(N.data=E)))),(E=ey?ty(e,n):ny(e,n))&&(u=Ia(u,"onBeforeInput"),0<u.length&&(c=new rc("onBeforeInput","beforeinput",null,n,c),d.push({event:c,listeners:u}),c.data=E))}lf(d,t)})}function yi(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Ia(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,a=i.stateNode;i.tag===5&&a!==null&&(i=a,a=ci(e,n),a!=null&&r.unshift(yi(e,a,i)),a=ci(e,t),a!=null&&r.push(yi(e,a,i))),e=e.return}return r}function Qn(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function hc(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,l=s.alternate,u=s.stateNode;if(l!==null&&l===r)break;s.tag===5&&u!==null&&(s=u,i?(l=ci(n,a),l!=null&&o.unshift(yi(n,l,s))):i||(l=ci(n,a),l!=null&&o.push(yi(n,l,s)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var gy=/\r\n?/g,yy=/\u0000|\uFFFD/g;function gc(e){return(typeof e=="string"?e:""+e).replace(gy,`
`).replace(yy,"")}function Gi(e,t,n){if(t=gc(t),gc(e)!==t&&n)throw Error(A(425))}function ja(){}var Bs=null,qs=null;function Us(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Vs=typeof setTimeout=="function"?setTimeout:void 0,xy=typeof clearTimeout=="function"?clearTimeout:void 0,yc=typeof Promise=="function"?Promise:void 0,_y=typeof queueMicrotask=="function"?queueMicrotask:typeof yc<"u"?function(e){return yc.resolve(null).then(e).catch(vy)}:Vs;function vy(e){setTimeout(function(){throw e})}function Ho(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),fi(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);fi(t)}function dn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function xc(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Er=Math.random().toString(36).slice(2),Rt="__reactFiber$"+Er,xi="__reactProps$"+Er,Xt="__reactContainer$"+Er,Hs="__reactEvents$"+Er,wy="__reactListeners$"+Er,ky="__reactHandles$"+Er;function Rn(e){var t=e[Rt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[Xt]||n[Rt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=xc(e);e!==null;){if(n=e[Rt])return n;e=xc(e)}return t}e=n,n=e.parentNode}return null}function Ri(e){return e=e[Rt]||e[Xt],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function er(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(A(33))}function uo(e){return e[xi]||null}var $s=[],tr=-1;function kn(e){return{current:e}}function fe(e){0>tr||(e.current=$s[tr],$s[tr]=null,tr--)}function ce(e,t){tr++,$s[tr]=e.current,e.current=t}var vn={},Me=kn(vn),Qe=kn(!1),Dn=vn;function yr(e,t){var n=e.type.contextTypes;if(!n)return vn;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},a;for(a in n)i[a]=t[a];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function Xe(e){return e=e.childContextTypes,e!=null}function Oa(){fe(Qe),fe(Me)}function _c(e,t,n){if(Me.current!==vn)throw Error(A(168));ce(Me,t),ce(Qe,n)}function cf(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(A(108,og(e)||"Unknown",i));return ye({},n,r)}function Ma(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||vn,Dn=Me.current,ce(Me,e),ce(Qe,Qe.current),!0}function vc(e,t,n){var r=e.stateNode;if(!r)throw Error(A(169));n?(e=cf(e,t,Dn),r.__reactInternalMemoizedMergedChildContext=e,fe(Qe),fe(Me),ce(Me,e)):fe(Qe),ce(Qe,n)}var Ut=null,co=!1,$o=!1;function df(e){Ut===null?Ut=[e]:Ut.push(e)}function by(e){co=!0,df(e)}function bn(){if(!$o&&Ut!==null){$o=!0;var e=0,t=oe;try{var n=Ut;for(oe=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}Ut=null,co=!1}catch(i){throw Ut!==null&&(Ut=Ut.slice(e+1)),jp(Dl,bn),i}finally{oe=t,$o=!1}}return null}var nr=[],rr=0,Da=null,Fa=0,ct=[],dt=0,Fn=null,Vt=1,Ht="";function Tn(e,t){nr[rr++]=Fa,nr[rr++]=Da,Da=e,Fa=t}function pf(e,t,n){ct[dt++]=Vt,ct[dt++]=Ht,ct[dt++]=Fn,Fn=e;var r=Vt;e=Ht;var i=32-zt(r)-1;r&=~(1<<i),n+=1;var a=32-zt(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Vt=1<<32-zt(t)+i|n<<i|r,Ht=a+e}else Vt=1<<a|n<<i|r,Ht=e}function Ql(e){e.return!==null&&(Tn(e,1),pf(e,1,0))}function Xl(e){for(;e===Da;)Da=nr[--rr],nr[rr]=null,Fa=nr[--rr],nr[rr]=null;for(;e===Fn;)Fn=ct[--dt],ct[dt]=null,Ht=ct[--dt],ct[dt]=null,Vt=ct[--dt],ct[dt]=null}var it=null,nt=null,me=!1,Ct=null;function ff(e,t){var n=ft(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function wc(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,it=e,nt=dn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,it=e,nt=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Fn!==null?{id:Vt,overflow:Ht}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=ft(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,it=e,nt=null,!0):!1;default:return!1}}function Qs(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Xs(e){if(me){var t=nt;if(t){var n=t;if(!wc(e,t)){if(Qs(e))throw Error(A(418));t=dn(n.nextSibling);var r=it;t&&wc(e,t)?ff(r,n):(e.flags=e.flags&-4097|2,me=!1,it=e)}}else{if(Qs(e))throw Error(A(418));e.flags=e.flags&-4097|2,me=!1,it=e}}}function kc(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;it=e}function Zi(e){if(e!==it)return!1;if(!me)return kc(e),me=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!Us(e.type,e.memoizedProps)),t&&(t=nt)){if(Qs(e))throw mf(),Error(A(418));for(;t;)ff(e,t),t=dn(t.nextSibling)}if(kc(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(A(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){nt=dn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}nt=null}}else nt=it?dn(e.stateNode.nextSibling):null;return!0}function mf(){for(var e=nt;e;)e=dn(e.nextSibling)}function xr(){nt=it=null,me=!1}function Kl(e){Ct===null?Ct=[e]:Ct.push(e)}var Sy=Gt.ReactCurrentBatchConfig;function Dr(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(A(309));var r=n.stateNode}if(!r)throw Error(A(147,e));var i=r,a=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===a?t.ref:(t=function(o){var s=i.refs;o===null?delete s[a]:s[a]=o},t._stringRef=a,t)}if(typeof e!="string")throw Error(A(284));if(!n._owner)throw Error(A(290,e))}return e}function Ji(e,t){throw e=Object.prototype.toString.call(t),Error(A(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function bc(e){var t=e._init;return t(e._payload)}function hf(e){function t(h,m){if(e){var g=h.deletions;g===null?(h.deletions=[m],h.flags|=16):g.push(m)}}function n(h,m){if(!e)return null;for(;m!==null;)t(h,m),m=m.sibling;return null}function r(h,m){for(h=new Map;m!==null;)m.key!==null?h.set(m.key,m):h.set(m.index,m),m=m.sibling;return h}function i(h,m){return h=hn(h,m),h.index=0,h.sibling=null,h}function a(h,m,g){return h.index=g,e?(g=h.alternate,g!==null?(g=g.index,g<m?(h.flags|=2,m):g):(h.flags|=2,m)):(h.flags|=1048576,m)}function o(h){return e&&h.alternate===null&&(h.flags|=2),h}function s(h,m,g,b){return m===null||m.tag!==6?(m=Jo(g,h.mode,b),m.return=h,m):(m=i(m,g),m.return=h,m)}function l(h,m,g,b){var z=g.type;return z===Yn?c(h,m,g.props.children,b,g.key):m!==null&&(m.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===en&&bc(z)===m.type)?(b=i(m,g.props),b.ref=Dr(h,m,g),b.return=h,b):(b=ka(g.type,g.key,g.props,null,h.mode,b),b.ref=Dr(h,m,g),b.return=h,b)}function u(h,m,g,b){return m===null||m.tag!==4||m.stateNode.containerInfo!==g.containerInfo||m.stateNode.implementation!==g.implementation?(m=es(g,h.mode,b),m.return=h,m):(m=i(m,g.children||[]),m.return=h,m)}function c(h,m,g,b,z){return m===null||m.tag!==7?(m=Mn(g,h.mode,b,z),m.return=h,m):(m=i(m,g),m.return=h,m)}function d(h,m,g){if(typeof m=="string"&&m!==""||typeof m=="number")return m=Jo(""+m,h.mode,g),m.return=h,m;if(typeof m=="object"&&m!==null){switch(m.$$typeof){case qi:return g=ka(m.type,m.key,m.props,null,h.mode,g),g.ref=Dr(h,null,m),g.return=h,g;case Kn:return m=es(m,h.mode,g),m.return=h,m;case en:var b=m._init;return d(h,b(m._payload),g)}if(Hr(m)||Rr(m))return m=Mn(m,h.mode,g,null),m.return=h,m;Ji(h,m)}return null}function p(h,m,g,b){var z=m!==null?m.key:null;if(typeof g=="string"&&g!==""||typeof g=="number")return z!==null?null:s(h,m,""+g,b);if(typeof g=="object"&&g!==null){switch(g.$$typeof){case qi:return g.key===z?l(h,m,g,b):null;case Kn:return g.key===z?u(h,m,g,b):null;case en:return z=g._init,p(h,m,z(g._payload),b)}if(Hr(g)||Rr(g))return z!==null?null:c(h,m,g,b,null);Ji(h,g)}return null}function f(h,m,g,b,z){if(typeof b=="string"&&b!==""||typeof b=="number")return h=h.get(g)||null,s(m,h,""+b,z);if(typeof b=="object"&&b!==null){switch(b.$$typeof){case qi:return h=h.get(b.key===null?g:b.key)||null,l(m,h,b,z);case Kn:return h=h.get(b.key===null?g:b.key)||null,u(m,h,b,z);case en:var k=b._init;return f(h,m,g,k(b._payload),z)}if(Hr(b)||Rr(b))return h=h.get(g)||null,c(m,h,b,z,null);Ji(m,b)}return null}function w(h,m,g,b){for(var z=null,k=null,E=m,N=m=0,O=null;E!==null&&N<g.length;N++){E.index>N?(O=E,E=null):O=E.sibling;var L=p(h,E,g[N],b);if(L===null){E===null&&(E=O);break}e&&E&&L.alternate===null&&t(h,E),m=a(L,m,N),k===null?z=L:k.sibling=L,k=L,E=O}if(N===g.length)return n(h,E),me&&Tn(h,N),z;if(E===null){for(;N<g.length;N++)E=d(h,g[N],b),E!==null&&(m=a(E,m,N),k===null?z=E:k.sibling=E,k=E);return me&&Tn(h,N),z}for(E=r(h,E);N<g.length;N++)O=f(E,h,N,g[N],b),O!==null&&(e&&O.alternate!==null&&E.delete(O.key===null?N:O.key),m=a(O,m,N),k===null?z=O:k.sibling=O,k=O);return e&&E.forEach(function(F){return t(h,F)}),me&&Tn(h,N),z}function _(h,m,g,b){var z=Rr(g);if(typeof z!="function")throw Error(A(150));if(g=z.call(g),g==null)throw Error(A(151));for(var k=z=null,E=m,N=m=0,O=null,L=g.next();E!==null&&!L.done;N++,L=g.next()){E.index>N?(O=E,E=null):O=E.sibling;var F=p(h,E,L.value,b);if(F===null){E===null&&(E=O);break}e&&E&&F.alternate===null&&t(h,E),m=a(F,m,N),k===null?z=F:k.sibling=F,k=F,E=O}if(L.done)return n(h,E),me&&Tn(h,N),z;if(E===null){for(;!L.done;N++,L=g.next())L=d(h,L.value,b),L!==null&&(m=a(L,m,N),k===null?z=L:k.sibling=L,k=L);return me&&Tn(h,N),z}for(E=r(h,E);!L.done;N++,L=g.next())L=f(E,h,N,L.value,b),L!==null&&(e&&L.alternate!==null&&E.delete(L.key===null?N:L.key),m=a(L,m,N),k===null?z=L:k.sibling=L,k=L);return e&&E.forEach(function(U){return t(h,U)}),me&&Tn(h,N),z}function S(h,m,g,b){if(typeof g=="object"&&g!==null&&g.type===Yn&&g.key===null&&(g=g.props.children),typeof g=="object"&&g!==null){switch(g.$$typeof){case qi:e:{for(var z=g.key,k=m;k!==null;){if(k.key===z){if(z=g.type,z===Yn){if(k.tag===7){n(h,k.sibling),m=i(k,g.props.children),m.return=h,h=m;break e}}else if(k.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===en&&bc(z)===k.type){n(h,k.sibling),m=i(k,g.props),m.ref=Dr(h,k,g),m.return=h,h=m;break e}n(h,k);break}else t(h,k);k=k.sibling}g.type===Yn?(m=Mn(g.props.children,h.mode,b,g.key),m.return=h,h=m):(b=ka(g.type,g.key,g.props,null,h.mode,b),b.ref=Dr(h,m,g),b.return=h,h=b)}return o(h);case Kn:e:{for(k=g.key;m!==null;){if(m.key===k)if(m.tag===4&&m.stateNode.containerInfo===g.containerInfo&&m.stateNode.implementation===g.implementation){n(h,m.sibling),m=i(m,g.children||[]),m.return=h,h=m;break e}else{n(h,m);break}else t(h,m);m=m.sibling}m=es(g,h.mode,b),m.return=h,h=m}return o(h);case en:return k=g._init,S(h,m,k(g._payload),b)}if(Hr(g))return w(h,m,g,b);if(Rr(g))return _(h,m,g,b);Ji(h,g)}return typeof g=="string"&&g!==""||typeof g=="number"?(g=""+g,m!==null&&m.tag===6?(n(h,m.sibling),m=i(m,g),m.return=h,h=m):(n(h,m),m=Jo(g,h.mode,b),m.return=h,h=m),o(h)):n(h,m)}return S}var _r=hf(!0),gf=hf(!1),Wa=kn(null),Ba=null,ir=null,Yl=null;function Gl(){Yl=ir=Ba=null}function Zl(e){var t=Wa.current;fe(Wa),e._currentValue=t}function Ks(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function pr(e,t){Ba=e,Yl=ir=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&($e=!0),e.firstContext=null)}function ht(e){var t=e._currentValue;if(Yl!==e)if(e={context:e,memoizedValue:t,next:null},ir===null){if(Ba===null)throw Error(A(308));ir=e,Ba.dependencies={lanes:0,firstContext:e}}else ir=ir.next=e;return t}var In=null;function Jl(e){In===null?In=[e]:In.push(e)}function yf(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,Jl(t)):(n.next=i.next,i.next=n),t.interleaved=n,Kt(e,r)}function Kt(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var tn=!1;function eu(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function xf(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function $t(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function pn(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,ne&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,Kt(e,n)}return i=r.interleaved,i===null?(t.next=t,Jl(r)):(t.next=i.next,i.next=t),r.interleaved=t,Kt(e,n)}function ga(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Fl(e,n)}}function Sc(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function qa(e,t,n,r){var i=e.updateQueue;tn=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var l=s,u=l.next;l.next=null,o===null?a=u:o.next=u,o=l;var c=e.alternate;c!==null&&(c=c.updateQueue,s=c.lastBaseUpdate,s!==o&&(s===null?c.firstBaseUpdate=u:s.next=u,c.lastBaseUpdate=l))}if(a!==null){var d=i.baseState;o=0,c=u=l=null,s=a;do{var p=s.lane,f=s.eventTime;if((r&p)===p){c!==null&&(c=c.next={eventTime:f,lane:0,tag:s.tag,payload:s.payload,callback:s.callback,next:null});e:{var w=e,_=s;switch(p=t,f=n,_.tag){case 1:if(w=_.payload,typeof w=="function"){d=w.call(f,d,p);break e}d=w;break e;case 3:w.flags=w.flags&-65537|128;case 0:if(w=_.payload,p=typeof w=="function"?w.call(f,d,p):w,p==null)break e;d=ye({},d,p);break e;case 2:tn=!0}}s.callback!==null&&s.lane!==0&&(e.flags|=64,p=i.effects,p===null?i.effects=[s]:p.push(s))}else f={eventTime:f,lane:p,tag:s.tag,payload:s.payload,callback:s.callback,next:null},c===null?(u=c=f,l=d):c=c.next=f,o|=p;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(!0);if(c===null&&(l=d),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=c,t=i.shared.interleaved,t!==null){i=t;do o|=i.lane,i=i.next;while(i!==t)}else a===null&&(i.shared.lanes=0);Bn|=o,e.lanes=o,e.memoizedState=d}}function Cc(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(A(191,i));i.call(r)}}}var Ii={},Mt=kn(Ii),_i=kn(Ii),vi=kn(Ii);function jn(e){if(e===Ii)throw Error(A(174));return e}function tu(e,t){switch(ce(vi,t),ce(_i,e),ce(Mt,Ii),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:Ns(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=Ns(t,e)}fe(Mt),ce(Mt,t)}function vr(){fe(Mt),fe(_i),fe(vi)}function _f(e){jn(vi.current);var t=jn(Mt.current),n=Ns(t,e.type);t!==n&&(ce(_i,e),ce(Mt,n))}function nu(e){_i.current===e&&(fe(Mt),fe(_i))}var he=kn(0);function Ua(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Qo=[];function ru(){for(var e=0;e<Qo.length;e++)Qo[e]._workInProgressVersionPrimary=null;Qo.length=0}var ya=Gt.ReactCurrentDispatcher,Xo=Gt.ReactCurrentBatchConfig,Wn=0,ge=null,ze=null,Pe=null,Va=!1,ei=!1,wi=0,Cy=0;function Re(){throw Error(A(321))}function iu(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Pt(e[n],t[n]))return!1;return!0}function au(e,t,n,r,i,a){if(Wn=a,ge=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,ya.current=e===null||e.memoizedState===null?Ny:Ty,e=n(r,i),ei){a=0;do{if(ei=!1,wi=0,25<=a)throw Error(A(301));a+=1,Pe=ze=null,t.updateQueue=null,ya.current=Ly,e=n(r,i)}while(ei)}if(ya.current=Ha,t=ze!==null&&ze.next!==null,Wn=0,Pe=ze=ge=null,Va=!1,t)throw Error(A(300));return e}function ou(){var e=wi!==0;return wi=0,e}function Lt(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Pe===null?ge.memoizedState=Pe=e:Pe=Pe.next=e,Pe}function gt(){if(ze===null){var e=ge.alternate;e=e!==null?e.memoizedState:null}else e=ze.next;var t=Pe===null?ge.memoizedState:Pe.next;if(t!==null)Pe=t,ze=e;else{if(e===null)throw Error(A(310));ze=e,e={memoizedState:ze.memoizedState,baseState:ze.baseState,baseQueue:ze.baseQueue,queue:ze.queue,next:null},Pe===null?ge.memoizedState=Pe=e:Pe=Pe.next=e}return Pe}function ki(e,t){return typeof t=="function"?t(e):t}function Ko(e){var t=gt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=ze,i=r.baseQueue,a=n.pending;if(a!==null){if(i!==null){var o=i.next;i.next=a.next,a.next=o}r.baseQueue=i=a,n.pending=null}if(i!==null){a=i.next,r=r.baseState;var s=o=null,l=null,u=a;do{var c=u.lane;if((Wn&c)===c)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var d={lane:c,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(s=l=d,o=r):l=l.next=d,ge.lanes|=c,Bn|=c}u=u.next}while(u!==null&&u!==a);l===null?o=r:l.next=s,Pt(r,t.memoizedState)||($e=!0),t.memoizedState=r,t.baseState=o,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do a=i.lane,ge.lanes|=a,Bn|=a,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function Yo(e){var t=gt(),n=t.queue;if(n===null)throw Error(A(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var o=i=i.next;do a=e(a,o.action),o=o.next;while(o!==i);Pt(a,t.memoizedState)||($e=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function vf(){}function wf(e,t){var n=ge,r=gt(),i=t(),a=!Pt(r.memoizedState,i);if(a&&(r.memoizedState=i,$e=!0),r=r.queue,su(Sf.bind(null,n,r,e),[e]),r.getSnapshot!==t||a||Pe!==null&&Pe.memoizedState.tag&1){if(n.flags|=2048,bi(9,bf.bind(null,n,r,i,t),void 0,null),Ne===null)throw Error(A(349));Wn&30||kf(n,t,i)}return i}function kf(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=ge.updateQueue,t===null?(t={lastEffect:null,stores:null},ge.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function bf(e,t,n,r){t.value=n,t.getSnapshot=r,Cf(t)&&zf(e)}function Sf(e,t,n){return n(function(){Cf(t)&&zf(e)})}function Cf(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Pt(e,n)}catch{return!0}}function zf(e){var t=Kt(e,1);t!==null&&Et(t,e,1,-1)}function zc(e){var t=Lt();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:ki,lastRenderedState:e},t.queue=e,e=e.dispatch=Py.bind(null,ge,e),[t.memoizedState,e]}function bi(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=ge.updateQueue,t===null?(t={lastEffect:null,stores:null},ge.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Ef(){return gt().memoizedState}function xa(e,t,n,r){var i=Lt();ge.flags|=e,i.memoizedState=bi(1|t,n,void 0,r===void 0?null:r)}function po(e,t,n,r){var i=gt();r=r===void 0?null:r;var a=void 0;if(ze!==null){var o=ze.memoizedState;if(a=o.destroy,r!==null&&iu(r,o.deps)){i.memoizedState=bi(t,n,a,r);return}}ge.flags|=e,i.memoizedState=bi(1|t,n,a,r)}function Ec(e,t){return xa(8390656,8,e,t)}function su(e,t){return po(2048,8,e,t)}function Pf(e,t){return po(4,2,e,t)}function Nf(e,t){return po(4,4,e,t)}function Tf(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Lf(e,t,n){return n=n!=null?n.concat([e]):null,po(4,4,Tf.bind(null,t,e),n)}function lu(){}function Af(e,t){var n=gt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&iu(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Rf(e,t){var n=gt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&iu(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function If(e,t,n){return Wn&21?(Pt(n,t)||(n=Dp(),ge.lanes|=n,Bn|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,$e=!0),e.memoizedState=n)}function zy(e,t){var n=oe;oe=n!==0&&4>n?n:4,e(!0);var r=Xo.transition;Xo.transition={};try{e(!1),t()}finally{oe=n,Xo.transition=r}}function jf(){return gt().memoizedState}function Ey(e,t,n){var r=mn(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},Of(e))Mf(t,n);else if(n=yf(e,t,n,r),n!==null){var i=Be();Et(n,e,r,i),Df(n,t,r)}}function Py(e,t,n){var r=mn(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(Of(e))Mf(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Pt(s,o)){var l=t.interleaved;l===null?(i.next=i,Jl(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=yf(e,t,i,r),n!==null&&(i=Be(),Et(n,e,r,i),Df(n,t,r))}}function Of(e){var t=e.alternate;return e===ge||t!==null&&t===ge}function Mf(e,t){ei=Va=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Df(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Fl(e,n)}}var Ha={readContext:ht,useCallback:Re,useContext:Re,useEffect:Re,useImperativeHandle:Re,useInsertionEffect:Re,useLayoutEffect:Re,useMemo:Re,useReducer:Re,useRef:Re,useState:Re,useDebugValue:Re,useDeferredValue:Re,useTransition:Re,useMutableSource:Re,useSyncExternalStore:Re,useId:Re,unstable_isNewReconciler:!1},Ny={readContext:ht,useCallback:function(e,t){return Lt().memoizedState=[e,t===void 0?null:t],e},useContext:ht,useEffect:Ec,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,xa(4194308,4,Tf.bind(null,t,e),n)},useLayoutEffect:function(e,t){return xa(4194308,4,e,t)},useInsertionEffect:function(e,t){return xa(4,2,e,t)},useMemo:function(e,t){var n=Lt();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=Lt();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=Ey.bind(null,ge,e),[r.memoizedState,e]},useRef:function(e){var t=Lt();return e={current:e},t.memoizedState=e},useState:zc,useDebugValue:lu,useDeferredValue:function(e){return Lt().memoizedState=e},useTransition:function(){var e=zc(!1),t=e[0];return e=zy.bind(null,e[1]),Lt().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=ge,i=Lt();if(me){if(n===void 0)throw Error(A(407));n=n()}else{if(n=t(),Ne===null)throw Error(A(349));Wn&30||kf(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,Ec(Sf.bind(null,r,a,e),[e]),r.flags|=2048,bi(9,bf.bind(null,r,a,n,t),void 0,null),n},useId:function(){var e=Lt(),t=Ne.identifierPrefix;if(me){var n=Ht,r=Vt;n=(r&~(1<<32-zt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=wi++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=Cy++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},Ty={readContext:ht,useCallback:Af,useContext:ht,useEffect:su,useImperativeHandle:Lf,useInsertionEffect:Pf,useLayoutEffect:Nf,useMemo:Rf,useReducer:Ko,useRef:Ef,useState:function(){return Ko(ki)},useDebugValue:lu,useDeferredValue:function(e){var t=gt();return If(t,ze.memoizedState,e)},useTransition:function(){var e=Ko(ki)[0],t=gt().memoizedState;return[e,t]},useMutableSource:vf,useSyncExternalStore:wf,useId:jf,unstable_isNewReconciler:!1},Ly={readContext:ht,useCallback:Af,useContext:ht,useEffect:su,useImperativeHandle:Lf,useInsertionEffect:Pf,useLayoutEffect:Nf,useMemo:Rf,useReducer:Yo,useRef:Ef,useState:function(){return Yo(ki)},useDebugValue:lu,useDeferredValue:function(e){var t=gt();return ze===null?t.memoizedState=e:If(t,ze.memoizedState,e)},useTransition:function(){var e=Yo(ki)[0],t=gt().memoizedState;return[e,t]},useMutableSource:vf,useSyncExternalStore:wf,useId:jf,unstable_isNewReconciler:!1};function bt(e,t){if(e&&e.defaultProps){t=ye({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function Ys(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:ye({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var fo={isMounted:function(e){return(e=e._reactInternals)?Vn(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=Be(),i=mn(e),a=$t(r,i);a.payload=t,n!=null&&(a.callback=n),t=pn(e,a,i),t!==null&&(Et(t,e,i,r),ga(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=Be(),i=mn(e),a=$t(r,i);a.tag=1,a.payload=t,n!=null&&(a.callback=n),t=pn(e,a,i),t!==null&&(Et(t,e,i,r),ga(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=Be(),r=mn(e),i=$t(n,r);i.tag=2,t!=null&&(i.callback=t),t=pn(e,i,r),t!==null&&(Et(t,e,r,n),ga(t,e,r))}};function Pc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!hi(n,r)||!hi(i,a):!0}function Ff(e,t,n){var r=!1,i=vn,a=t.contextType;return typeof a=="object"&&a!==null?a=ht(a):(i=Xe(t)?Dn:Me.current,r=t.contextTypes,a=(r=r!=null)?yr(e,i):vn),t=new t(n,a),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=fo,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=a),t}function Nc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&fo.enqueueReplaceState(t,t.state,null)}function Gs(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},eu(e);var a=t.contextType;typeof a=="object"&&a!==null?i.context=ht(a):(a=Xe(t)?Dn:Me.current,i.context=yr(e,a)),i.state=e.memoizedState,a=t.getDerivedStateFromProps,typeof a=="function"&&(Ys(e,t,a,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&fo.enqueueReplaceState(i,i.state,null),qa(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function wr(e,t){try{var n="",r=t;do n+=ag(r),r=r.return;while(r);var i=n}catch(a){i=`
Error generating stack: `+a.message+`
`+a.stack}return{value:e,source:t,stack:i,digest:null}}function Go(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function Zs(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var Ay=typeof WeakMap=="function"?WeakMap:Map;function Wf(e,t,n){n=$t(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){Qa||(Qa=!0,ll=r),Zs(e,t)},n}function Bf(e,t,n){n=$t(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){Zs(e,t)}}var a=e.stateNode;return a!==null&&typeof a.componentDidCatch=="function"&&(n.callback=function(){Zs(e,t),typeof r!="function"&&(fn===null?fn=new Set([this]):fn.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function Tc(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Ay;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=$y.bind(null,e,t,n),t.then(e,e))}function Lc(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Ac(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=$t(-1,1),t.tag=2,pn(n,t,1))),n.lanes|=1),e)}var Ry=Gt.ReactCurrentOwner,$e=!1;function We(e,t,n,r){t.child=e===null?gf(t,null,n,r):_r(t,e.child,n,r)}function Rc(e,t,n,r,i){n=n.render;var a=t.ref;return pr(t,i),r=au(e,t,n,r,a,i),n=ou(),e!==null&&!$e?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,Yt(e,t,i)):(me&&n&&Ql(t),t.flags|=1,We(e,t,r,i),t.child)}function Ic(e,t,n,r,i){if(e===null){var a=n.type;return typeof a=="function"&&!gu(a)&&a.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=a,qf(e,t,a,r,i)):(e=ka(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!(e.lanes&i)){var o=a.memoizedProps;if(n=n.compare,n=n!==null?n:hi,n(o,r)&&e.ref===t.ref)return Yt(e,t,i)}return t.flags|=1,e=hn(a,r),e.ref=t.ref,e.return=t,t.child=e}function qf(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(hi(a,r)&&e.ref===t.ref)if($e=!1,t.pendingProps=r=a,(e.lanes&i)!==0)e.flags&131072&&($e=!0);else return t.lanes=e.lanes,Yt(e,t,i)}return Js(e,t,n,r,i)}function Uf(e,t,n){var r=t.pendingProps,i=r.children,a=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},ce(or,et),et|=n;else{if(!(n&1073741824))return e=a!==null?a.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,ce(or,et),et|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=a!==null?a.baseLanes:n,ce(or,et),et|=r}else a!==null?(r=a.baseLanes|n,t.memoizedState=null):r=n,ce(or,et),et|=r;return We(e,t,i,n),t.child}function Vf(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function Js(e,t,n,r,i){var a=Xe(n)?Dn:Me.current;return a=yr(t,a),pr(t,i),n=au(e,t,n,r,a,i),r=ou(),e!==null&&!$e?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,Yt(e,t,i)):(me&&r&&Ql(t),t.flags|=1,We(e,t,n,i),t.child)}function jc(e,t,n,r,i){if(Xe(n)){var a=!0;Ma(t)}else a=!1;if(pr(t,i),t.stateNode===null)_a(e,t),Ff(t,n,r),Gs(t,n,r,i),r=!0;else if(e===null){var o=t.stateNode,s=t.memoizedProps;o.props=s;var l=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=ht(u):(u=Xe(n)?Dn:Me.current,u=yr(t,u));var c=n.getDerivedStateFromProps,d=typeof c=="function"||typeof o.getSnapshotBeforeUpdate=="function";d||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==r||l!==u)&&Nc(t,o,r,u),tn=!1;var p=t.memoizedState;o.state=p,qa(t,r,o,i),l=t.memoizedState,s!==r||p!==l||Qe.current||tn?(typeof c=="function"&&(Ys(t,n,c,r),l=t.memoizedState),(s=tn||Pc(t,n,s,r,p,l,u))?(d||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),o.props=r,o.state=l,o.context=u,r=s):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{o=t.stateNode,xf(e,t),s=t.memoizedProps,u=t.type===t.elementType?s:bt(t.type,s),o.props=u,d=t.pendingProps,p=o.context,l=n.contextType,typeof l=="object"&&l!==null?l=ht(l):(l=Xe(n)?Dn:Me.current,l=yr(t,l));var f=n.getDerivedStateFromProps;(c=typeof f=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(s!==d||p!==l)&&Nc(t,o,r,l),tn=!1,p=t.memoizedState,o.state=p,qa(t,r,o,i);var w=t.memoizedState;s!==d||p!==w||Qe.current||tn?(typeof f=="function"&&(Ys(t,n,f,r),w=t.memoizedState),(u=tn||Pc(t,n,u,r,p,w,l)||!1)?(c||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(r,w,l),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(r,w,l)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=w),o.props=r,o.state=w,o.context=l,r=u):(typeof o.componentDidUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&p===e.memoizedState||(t.flags|=1024),r=!1)}return el(e,t,n,r,a,i)}function el(e,t,n,r,i,a){Vf(e,t);var o=(t.flags&128)!==0;if(!r&&!o)return i&&vc(t,n,!1),Yt(e,t,a);r=t.stateNode,Ry.current=t;var s=o&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&o?(t.child=_r(t,e.child,null,a),t.child=_r(t,null,s,a)):We(e,t,s,a),t.memoizedState=r.state,i&&vc(t,n,!0),t.child}function Hf(e){var t=e.stateNode;t.pendingContext?_c(e,t.pendingContext,t.pendingContext!==t.context):t.context&&_c(e,t.context,!1),tu(e,t.containerInfo)}function Oc(e,t,n,r,i){return xr(),Kl(i),t.flags|=256,We(e,t,n,r),t.child}var tl={dehydrated:null,treeContext:null,retryLane:0};function nl(e){return{baseLanes:e,cachePool:null,transitions:null}}function $f(e,t,n){var r=t.pendingProps,i=he.current,a=!1,o=(t.flags&128)!==0,s;if((s=o)||(s=e!==null&&e.memoizedState===null?!1:(i&2)!==0),s?(a=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),ce(he,i&1),e===null)return Xs(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=r.children,e=r.fallback,a?(r=t.mode,a=t.child,o={mode:"hidden",children:o},!(r&1)&&a!==null?(a.childLanes=0,a.pendingProps=o):a=go(o,r,0,null),e=Mn(e,r,n,null),a.return=t,e.return=t,a.sibling=e,t.child=a,t.child.memoizedState=nl(n),t.memoizedState=tl,e):uu(t,o));if(i=e.memoizedState,i!==null&&(s=i.dehydrated,s!==null))return Iy(e,t,o,r,s,i,n);if(a){a=r.fallback,o=t.mode,i=e.child,s=i.sibling;var l={mode:"hidden",children:r.children};return!(o&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=hn(i,l),r.subtreeFlags=i.subtreeFlags&14680064),s!==null?a=hn(s,a):(a=Mn(a,o,n,null),a.flags|=2),a.return=t,r.return=t,r.sibling=a,t.child=r,r=a,a=t.child,o=e.child.memoizedState,o=o===null?nl(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},a.memoizedState=o,a.childLanes=e.childLanes&~n,t.memoizedState=tl,r}return a=e.child,e=a.sibling,r=hn(a,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function uu(e,t){return t=go({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function ea(e,t,n,r){return r!==null&&Kl(r),_r(t,e.child,null,n),e=uu(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function Iy(e,t,n,r,i,a,o){if(n)return t.flags&256?(t.flags&=-257,r=Go(Error(A(422))),ea(e,t,o,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(a=r.fallback,i=t.mode,r=go({mode:"visible",children:r.children},i,0,null),a=Mn(a,i,o,null),a.flags|=2,r.return=t,a.return=t,r.sibling=a,t.child=r,t.mode&1&&_r(t,e.child,null,o),t.child.memoizedState=nl(o),t.memoizedState=tl,a);if(!(t.mode&1))return ea(e,t,o,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var s=r.dgst;return r=s,a=Error(A(419)),r=Go(a,r,void 0),ea(e,t,o,r)}if(s=(o&e.childLanes)!==0,$e||s){if(r=Ne,r!==null){switch(o&-o){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|o)?0:i,i!==0&&i!==a.retryLane&&(a.retryLane=i,Kt(e,i),Et(r,e,i,-1))}return hu(),r=Go(Error(A(421))),ea(e,t,o,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=Qy.bind(null,e),i._reactRetry=t,null):(e=a.treeContext,nt=dn(i.nextSibling),it=t,me=!0,Ct=null,e!==null&&(ct[dt++]=Vt,ct[dt++]=Ht,ct[dt++]=Fn,Vt=e.id,Ht=e.overflow,Fn=t),t=uu(t,r.children),t.flags|=4096,t)}function Mc(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Ks(e.return,t,n)}function Zo(e,t,n,r,i){var a=e.memoizedState;a===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(a.isBackwards=t,a.rendering=null,a.renderingStartTime=0,a.last=r,a.tail=n,a.tailMode=i)}function Qf(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;if(We(e,t,r.children,n),r=he.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&Mc(e,n,t);else if(e.tag===19)Mc(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(ce(he,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&Ua(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),Zo(t,!1,i,n,a);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Ua(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}Zo(t,!0,n,null,a);break;case"together":Zo(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function _a(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function Yt(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Bn|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(A(153));if(t.child!==null){for(e=t.child,n=hn(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=hn(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function jy(e,t,n){switch(t.tag){case 3:Hf(t),xr();break;case 5:_f(t);break;case 1:Xe(t.type)&&Ma(t);break;case 4:tu(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;ce(Wa,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(ce(he,he.current&1),t.flags|=128,null):n&t.child.childLanes?$f(e,t,n):(ce(he,he.current&1),e=Yt(e,t,n),e!==null?e.sibling:null);ce(he,he.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return Qf(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),ce(he,he.current),r)break;return null;case 22:case 23:return t.lanes=0,Uf(e,t,n)}return Yt(e,t,n)}var Xf,rl,Kf,Yf;Xf=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};rl=function(){};Kf=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,jn(Mt.current);var a=null;switch(n){case"input":i=Cs(e,i),r=Cs(e,r),a=[];break;case"select":i=ye({},i,{value:void 0}),r=ye({},r,{value:void 0}),a=[];break;case"textarea":i=Ps(e,i),r=Ps(e,r),a=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=ja)}Ts(n,r);var o;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var s=i[u];for(o in s)s.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(li.hasOwnProperty(u)?a||(a=[]):(a=a||[]).push(u,null));for(u in r){var l=r[u];if(s=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==s&&(l!=null||s!=null))if(u==="style")if(s){for(o in s)!s.hasOwnProperty(o)||l&&l.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in l)l.hasOwnProperty(o)&&s[o]!==l[o]&&(n||(n={}),n[o]=l[o])}else n||(a||(a=[]),a.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,s=s?s.__html:void 0,l!=null&&s!==l&&(a=a||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(a=a||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(li.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&pe("scroll",e),a||s===l||(a=[])):(a=a||[]).push(u,l))}n&&(a=a||[]).push("style",n);var u=a;(t.updateQueue=u)&&(t.flags|=4)}};Yf=function(e,t,n,r){n!==r&&(t.flags|=4)};function Fr(e,t){if(!me)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function Ie(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function Oy(e,t,n){var r=t.pendingProps;switch(Xl(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Ie(t),null;case 1:return Xe(t.type)&&Oa(),Ie(t),null;case 3:return r=t.stateNode,vr(),fe(Qe),fe(Me),ru(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(Zi(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,Ct!==null&&(dl(Ct),Ct=null))),rl(e,t),Ie(t),null;case 5:nu(t);var i=jn(vi.current);if(n=t.type,e!==null&&t.stateNode!=null)Kf(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(A(166));return Ie(t),null}if(e=jn(Mt.current),Zi(t)){r=t.stateNode,n=t.type;var a=t.memoizedProps;switch(r[Rt]=t,r[xi]=a,e=(t.mode&1)!==0,n){case"dialog":pe("cancel",r),pe("close",r);break;case"iframe":case"object":case"embed":pe("load",r);break;case"video":case"audio":for(i=0;i<Qr.length;i++)pe(Qr[i],r);break;case"source":pe("error",r);break;case"img":case"image":case"link":pe("error",r),pe("load",r);break;case"details":pe("toggle",r);break;case"input":$u(r,a),pe("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!a.multiple},pe("invalid",r);break;case"textarea":Xu(r,a),pe("invalid",r)}Ts(n,a),i=null;for(var o in a)if(a.hasOwnProperty(o)){var s=a[o];o==="children"?typeof s=="string"?r.textContent!==s&&(a.suppressHydrationWarning!==!0&&Gi(r.textContent,s,e),i=["children",s]):typeof s=="number"&&r.textContent!==""+s&&(a.suppressHydrationWarning!==!0&&Gi(r.textContent,s,e),i=["children",""+s]):li.hasOwnProperty(o)&&s!=null&&o==="onScroll"&&pe("scroll",r)}switch(n){case"input":Ui(r),Qu(r,a,!0);break;case"textarea":Ui(r),Ku(r);break;case"select":case"option":break;default:typeof a.onClick=="function"&&(r.onclick=ja)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{o=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=bp(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=o.createElement(n,{is:r.is}):(e=o.createElement(n),n==="select"&&(o=e,r.multiple?o.multiple=!0:r.size&&(o.size=r.size))):e=o.createElementNS(e,n),e[Rt]=t,e[xi]=r,Xf(e,t,!1,!1),t.stateNode=e;e:{switch(o=Ls(n,r),n){case"dialog":pe("cancel",e),pe("close",e),i=r;break;case"iframe":case"object":case"embed":pe("load",e),i=r;break;case"video":case"audio":for(i=0;i<Qr.length;i++)pe(Qr[i],e);i=r;break;case"source":pe("error",e),i=r;break;case"img":case"image":case"link":pe("error",e),pe("load",e),i=r;break;case"details":pe("toggle",e),i=r;break;case"input":$u(e,r),i=Cs(e,r),pe("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=ye({},r,{value:void 0}),pe("invalid",e);break;case"textarea":Xu(e,r),i=Ps(e,r),pe("invalid",e);break;default:i=r}Ts(n,i),s=i;for(a in s)if(s.hasOwnProperty(a)){var l=s[a];a==="style"?zp(e,l):a==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&Sp(e,l)):a==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&ui(e,l):typeof l=="number"&&ui(e,""+l):a!=="suppressContentEditableWarning"&&a!=="suppressHydrationWarning"&&a!=="autoFocus"&&(li.hasOwnProperty(a)?l!=null&&a==="onScroll"&&pe("scroll",e):l!=null&&Rl(e,a,l,o))}switch(n){case"input":Ui(e),Qu(e,r,!1);break;case"textarea":Ui(e),Ku(e);break;case"option":r.value!=null&&e.setAttribute("value",""+_n(r.value));break;case"select":e.multiple=!!r.multiple,a=r.value,a!=null?lr(e,!!r.multiple,a,!1):r.defaultValue!=null&&lr(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=ja)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Ie(t),null;case 6:if(e&&t.stateNode!=null)Yf(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(A(166));if(n=jn(vi.current),jn(Mt.current),Zi(t)){if(r=t.stateNode,n=t.memoizedProps,r[Rt]=t,(a=r.nodeValue!==n)&&(e=it,e!==null))switch(e.tag){case 3:Gi(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&Gi(r.nodeValue,n,(e.mode&1)!==0)}a&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[Rt]=t,t.stateNode=r}return Ie(t),null;case 13:if(fe(he),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(me&&nt!==null&&t.mode&1&&!(t.flags&128))mf(),xr(),t.flags|=98560,a=!1;else if(a=Zi(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(A(318));if(a=t.memoizedState,a=a!==null?a.dehydrated:null,!a)throw Error(A(317));a[Rt]=t}else xr(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Ie(t),a=!1}else Ct!==null&&(dl(Ct),Ct=null),a=!0;if(!a)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||he.current&1?Ee===0&&(Ee=3):hu())),t.updateQueue!==null&&(t.flags|=4),Ie(t),null);case 4:return vr(),rl(e,t),e===null&&gi(t.stateNode.containerInfo),Ie(t),null;case 10:return Zl(t.type._context),Ie(t),null;case 17:return Xe(t.type)&&Oa(),Ie(t),null;case 19:if(fe(he),a=t.memoizedState,a===null)return Ie(t),null;if(r=(t.flags&128)!==0,o=a.rendering,o===null)if(r)Fr(a,!1);else{if(Ee!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=Ua(e),o!==null){for(t.flags|=128,Fr(a,!1),r=o.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)a=n,e=r,a.flags&=14680066,o=a.alternate,o===null?(a.childLanes=0,a.lanes=e,a.child=null,a.subtreeFlags=0,a.memoizedProps=null,a.memoizedState=null,a.updateQueue=null,a.dependencies=null,a.stateNode=null):(a.childLanes=o.childLanes,a.lanes=o.lanes,a.child=o.child,a.subtreeFlags=0,a.deletions=null,a.memoizedProps=o.memoizedProps,a.memoizedState=o.memoizedState,a.updateQueue=o.updateQueue,a.type=o.type,e=o.dependencies,a.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return ce(he,he.current&1|2),t.child}e=e.sibling}a.tail!==null&&ke()>kr&&(t.flags|=128,r=!0,Fr(a,!1),t.lanes=4194304)}else{if(!r)if(e=Ua(o),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),Fr(a,!0),a.tail===null&&a.tailMode==="hidden"&&!o.alternate&&!me)return Ie(t),null}else 2*ke()-a.renderingStartTime>kr&&n!==1073741824&&(t.flags|=128,r=!0,Fr(a,!1),t.lanes=4194304);a.isBackwards?(o.sibling=t.child,t.child=o):(n=a.last,n!==null?n.sibling=o:t.child=o,a.last=o)}return a.tail!==null?(t=a.tail,a.rendering=t,a.tail=t.sibling,a.renderingStartTime=ke(),t.sibling=null,n=he.current,ce(he,r?n&1|2:n&1),t):(Ie(t),null);case 22:case 23:return mu(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?et&1073741824&&(Ie(t),t.subtreeFlags&6&&(t.flags|=8192)):Ie(t),null;case 24:return null;case 25:return null}throw Error(A(156,t.tag))}function My(e,t){switch(Xl(t),t.tag){case 1:return Xe(t.type)&&Oa(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return vr(),fe(Qe),fe(Me),ru(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return nu(t),null;case 13:if(fe(he),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(A(340));xr()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return fe(he),null;case 4:return vr(),null;case 10:return Zl(t.type._context),null;case 22:case 23:return mu(),null;case 24:return null;default:return null}}var ta=!1,Oe=!1,Dy=typeof WeakSet=="function"?WeakSet:Set,q=null;function ar(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){_e(e,t,r)}else n.current=null}function il(e,t,n){try{n()}catch(r){_e(e,t,r)}}var Dc=!1;function Fy(e,t){if(Bs=Aa,e=ef(),$l(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,a=r.focusNode;r=r.focusOffset;try{n.nodeType,a.nodeType}catch{n=null;break e}var o=0,s=-1,l=-1,u=0,c=0,d=e,p=null;t:for(;;){for(var f;d!==n||i!==0&&d.nodeType!==3||(s=o+i),d!==a||r!==0&&d.nodeType!==3||(l=o+r),d.nodeType===3&&(o+=d.nodeValue.length),(f=d.firstChild)!==null;)p=d,d=f;for(;;){if(d===e)break t;if(p===n&&++u===i&&(s=o),p===a&&++c===r&&(l=o),(f=d.nextSibling)!==null)break;d=p,p=d.parentNode}d=f}n=s===-1||l===-1?null:{start:s,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(qs={focusedElem:e,selectionRange:n},Aa=!1,q=t;q!==null;)if(t=q,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,q=e;else for(;q!==null;){t=q;try{var w=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(w!==null){var _=w.memoizedProps,S=w.memoizedState,h=t.stateNode,m=h.getSnapshotBeforeUpdate(t.elementType===t.type?_:bt(t.type,_),S);h.__reactInternalSnapshotBeforeUpdate=m}break;case 3:var g=t.stateNode.containerInfo;g.nodeType===1?g.textContent="":g.nodeType===9&&g.documentElement&&g.removeChild(g.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(A(163))}}catch(b){_e(t,t.return,b)}if(e=t.sibling,e!==null){e.return=t.return,q=e;break}q=t.return}return w=Dc,Dc=!1,w}function ti(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var a=i.destroy;i.destroy=void 0,a!==void 0&&il(t,n,a)}i=i.next}while(i!==r)}}function mo(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function al(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function Gf(e){var t=e.alternate;t!==null&&(e.alternate=null,Gf(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[Rt],delete t[xi],delete t[Hs],delete t[wy],delete t[ky])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function Zf(e){return e.tag===5||e.tag===3||e.tag===4}function Fc(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||Zf(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function ol(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=ja));else if(r!==4&&(e=e.child,e!==null))for(ol(e,t,n),e=e.sibling;e!==null;)ol(e,t,n),e=e.sibling}function sl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(sl(e,t,n),e=e.sibling;e!==null;)sl(e,t,n),e=e.sibling}var Te=null,St=!1;function Zt(e,t,n){for(n=n.child;n!==null;)Jf(e,t,n),n=n.sibling}function Jf(e,t,n){if(Ot&&typeof Ot.onCommitFiberUnmount=="function")try{Ot.onCommitFiberUnmount(ao,n)}catch{}switch(n.tag){case 5:Oe||ar(n,t);case 6:var r=Te,i=St;Te=null,Zt(e,t,n),Te=r,St=i,Te!==null&&(St?(e=Te,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Te.removeChild(n.stateNode));break;case 18:Te!==null&&(St?(e=Te,n=n.stateNode,e.nodeType===8?Ho(e.parentNode,n):e.nodeType===1&&Ho(e,n),fi(e)):Ho(Te,n.stateNode));break;case 4:r=Te,i=St,Te=n.stateNode.containerInfo,St=!0,Zt(e,t,n),Te=r,St=i;break;case 0:case 11:case 14:case 15:if(!Oe&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var a=i,o=a.destroy;a=a.tag,o!==void 0&&(a&2||a&4)&&il(n,t,o),i=i.next}while(i!==r)}Zt(e,t,n);break;case 1:if(!Oe&&(ar(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(s){_e(n,t,s)}Zt(e,t,n);break;case 21:Zt(e,t,n);break;case 22:n.mode&1?(Oe=(r=Oe)||n.memoizedState!==null,Zt(e,t,n),Oe=r):Zt(e,t,n);break;default:Zt(e,t,n)}}function Wc(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Dy),t.forEach(function(r){var i=Xy.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function xt(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var a=e,o=t,s=o;e:for(;s!==null;){switch(s.tag){case 5:Te=s.stateNode,St=!1;break e;case 3:Te=s.stateNode.containerInfo,St=!0;break e;case 4:Te=s.stateNode.containerInfo,St=!0;break e}s=s.return}if(Te===null)throw Error(A(160));Jf(a,o,i),Te=null,St=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){_e(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)em(t,e),t=t.sibling}function em(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(xt(t,e),Tt(e),r&4){try{ti(3,e,e.return),mo(3,e)}catch(_){_e(e,e.return,_)}try{ti(5,e,e.return)}catch(_){_e(e,e.return,_)}}break;case 1:xt(t,e),Tt(e),r&512&&n!==null&&ar(n,n.return);break;case 5:if(xt(t,e),Tt(e),r&512&&n!==null&&ar(n,n.return),e.flags&32){var i=e.stateNode;try{ui(i,"")}catch(_){_e(e,e.return,_)}}if(r&4&&(i=e.stateNode,i!=null)){var a=e.memoizedProps,o=n!==null?n.memoizedProps:a,s=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{s==="input"&&a.type==="radio"&&a.name!=null&&wp(i,a),Ls(s,o);var u=Ls(s,a);for(o=0;o<l.length;o+=2){var c=l[o],d=l[o+1];c==="style"?zp(i,d):c==="dangerouslySetInnerHTML"?Sp(i,d):c==="children"?ui(i,d):Rl(i,c,d,u)}switch(s){case"input":zs(i,a);break;case"textarea":kp(i,a);break;case"select":var p=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!a.multiple;var f=a.value;f!=null?lr(i,!!a.multiple,f,!1):p!==!!a.multiple&&(a.defaultValue!=null?lr(i,!!a.multiple,a.defaultValue,!0):lr(i,!!a.multiple,a.multiple?[]:"",!1))}i[xi]=a}catch(_){_e(e,e.return,_)}}break;case 6:if(xt(t,e),Tt(e),r&4){if(e.stateNode===null)throw Error(A(162));i=e.stateNode,a=e.memoizedProps;try{i.nodeValue=a}catch(_){_e(e,e.return,_)}}break;case 3:if(xt(t,e),Tt(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{fi(t.containerInfo)}catch(_){_e(e,e.return,_)}break;case 4:xt(t,e),Tt(e);break;case 13:xt(t,e),Tt(e),i=e.child,i.flags&8192&&(a=i.memoizedState!==null,i.stateNode.isHidden=a,!a||i.alternate!==null&&i.alternate.memoizedState!==null||(pu=ke())),r&4&&Wc(e);break;case 22:if(c=n!==null&&n.memoizedState!==null,e.mode&1?(Oe=(u=Oe)||c,xt(t,e),Oe=u):xt(t,e),Tt(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!c&&e.mode&1)for(q=e,c=e.child;c!==null;){for(d=q=c;q!==null;){switch(p=q,f=p.child,p.tag){case 0:case 11:case 14:case 15:ti(4,p,p.return);break;case 1:ar(p,p.return);var w=p.stateNode;if(typeof w.componentWillUnmount=="function"){r=p,n=p.return;try{t=r,w.props=t.memoizedProps,w.state=t.memoizedState,w.componentWillUnmount()}catch(_){_e(r,n,_)}}break;case 5:ar(p,p.return);break;case 22:if(p.memoizedState!==null){qc(d);continue}}f!==null?(f.return=p,q=f):qc(d)}c=c.sibling}e:for(c=null,d=e;;){if(d.tag===5){if(c===null){c=d;try{i=d.stateNode,u?(a=i.style,typeof a.setProperty=="function"?a.setProperty("display","none","important"):a.display="none"):(s=d.stateNode,l=d.memoizedProps.style,o=l!=null&&l.hasOwnProperty("display")?l.display:null,s.style.display=Cp("display",o))}catch(_){_e(e,e.return,_)}}}else if(d.tag===6){if(c===null)try{d.stateNode.nodeValue=u?"":d.memoizedProps}catch(_){_e(e,e.return,_)}}else if((d.tag!==22&&d.tag!==23||d.memoizedState===null||d===e)&&d.child!==null){d.child.return=d,d=d.child;continue}if(d===e)break e;for(;d.sibling===null;){if(d.return===null||d.return===e)break e;c===d&&(c=null),d=d.return}c===d&&(c=null),d.sibling.return=d.return,d=d.sibling}}break;case 19:xt(t,e),Tt(e),r&4&&Wc(e);break;case 21:break;default:xt(t,e),Tt(e)}}function Tt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(Zf(n)){var r=n;break e}n=n.return}throw Error(A(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(ui(i,""),r.flags&=-33);var a=Fc(e);sl(e,a,i);break;case 3:case 4:var o=r.stateNode.containerInfo,s=Fc(e);ol(e,s,o);break;default:throw Error(A(161))}}catch(l){_e(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Wy(e,t,n){q=e,tm(e)}function tm(e,t,n){for(var r=(e.mode&1)!==0;q!==null;){var i=q,a=i.child;if(i.tag===22&&r){var o=i.memoizedState!==null||ta;if(!o){var s=i.alternate,l=s!==null&&s.memoizedState!==null||Oe;s=ta;var u=Oe;if(ta=o,(Oe=l)&&!u)for(q=i;q!==null;)o=q,l=o.child,o.tag===22&&o.memoizedState!==null?Uc(i):l!==null?(l.return=o,q=l):Uc(i);for(;a!==null;)q=a,tm(a),a=a.sibling;q=i,ta=s,Oe=u}Bc(e)}else i.subtreeFlags&8772&&a!==null?(a.return=i,q=a):Bc(e)}}function Bc(e){for(;q!==null;){var t=q;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Oe||mo(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!Oe)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:bt(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var a=t.updateQueue;a!==null&&Cc(t,a,r);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Cc(t,o,n)}break;case 5:var s=t.stateNode;if(n===null&&t.flags&4){n=s;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var c=u.memoizedState;if(c!==null){var d=c.dehydrated;d!==null&&fi(d)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(A(163))}Oe||t.flags&512&&al(t)}catch(p){_e(t,t.return,p)}}if(t===e){q=null;break}if(n=t.sibling,n!==null){n.return=t.return,q=n;break}q=t.return}}function qc(e){for(;q!==null;){var t=q;if(t===e){q=null;break}var n=t.sibling;if(n!==null){n.return=t.return,q=n;break}q=t.return}}function Uc(e){for(;q!==null;){var t=q;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{mo(4,t)}catch(l){_e(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){_e(t,i,l)}}var a=t.return;try{al(t)}catch(l){_e(t,a,l)}break;case 5:var o=t.return;try{al(t)}catch(l){_e(t,o,l)}}}catch(l){_e(t,t.return,l)}if(t===e){q=null;break}var s=t.sibling;if(s!==null){s.return=t.return,q=s;break}q=t.return}}var By=Math.ceil,$a=Gt.ReactCurrentDispatcher,cu=Gt.ReactCurrentOwner,mt=Gt.ReactCurrentBatchConfig,ne=0,Ne=null,Se=null,Le=0,et=0,or=kn(0),Ee=0,Si=null,Bn=0,ho=0,du=0,ni=null,He=null,pu=0,kr=1/0,qt=null,Qa=!1,ll=null,fn=null,na=!1,on=null,Xa=0,ri=0,ul=null,va=-1,wa=0;function Be(){return ne&6?ke():va!==-1?va:va=ke()}function mn(e){return e.mode&1?ne&2&&Le!==0?Le&-Le:Sy.transition!==null?(wa===0&&(wa=Dp()),wa):(e=oe,e!==0||(e=window.event,e=e===void 0?16:Hp(e.type)),e):1}function Et(e,t,n,r){if(50<ri)throw ri=0,ul=null,Error(A(185));Li(e,n,r),(!(ne&2)||e!==Ne)&&(e===Ne&&(!(ne&2)&&(ho|=n),Ee===4&&rn(e,Le)),Ke(e,r),n===1&&ne===0&&!(t.mode&1)&&(kr=ke()+500,co&&bn()))}function Ke(e,t){var n=e.callbackNode;Sg(e,t);var r=La(e,e===Ne?Le:0);if(r===0)n!==null&&Zu(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&Zu(n),t===1)e.tag===0?by(Vc.bind(null,e)):df(Vc.bind(null,e)),_y(function(){!(ne&6)&&bn()}),n=null;else{switch(Fp(r)){case 1:n=Dl;break;case 4:n=Op;break;case 16:n=Ta;break;case 536870912:n=Mp;break;default:n=Ta}n=um(n,nm.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function nm(e,t){if(va=-1,wa=0,ne&6)throw Error(A(327));var n=e.callbackNode;if(fr()&&e.callbackNode!==n)return null;var r=La(e,e===Ne?Le:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=Ka(e,r);else{t=r;var i=ne;ne|=2;var a=im();(Ne!==e||Le!==t)&&(qt=null,kr=ke()+500,On(e,t));do try{Vy();break}catch(s){rm(e,s)}while(!0);Gl(),$a.current=a,ne=i,Se!==null?t=0:(Ne=null,Le=0,t=Ee)}if(t!==0){if(t===2&&(i=Os(e),i!==0&&(r=i,t=cl(e,i))),t===1)throw n=Si,On(e,0),rn(e,r),Ke(e,ke()),n;if(t===6)rn(e,r);else{if(i=e.current.alternate,!(r&30)&&!qy(i)&&(t=Ka(e,r),t===2&&(a=Os(e),a!==0&&(r=a,t=cl(e,a))),t===1))throw n=Si,On(e,0),rn(e,r),Ke(e,ke()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(A(345));case 2:Ln(e,He,qt);break;case 3:if(rn(e,r),(r&130023424)===r&&(t=pu+500-ke(),10<t)){if(La(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){Be(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=Vs(Ln.bind(null,e,He,qt),t);break}Ln(e,He,qt);break;case 4:if(rn(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var o=31-zt(r);a=1<<o,o=t[o],o>i&&(i=o),r&=~a}if(r=i,r=ke()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*By(r/1960))-r,10<r){e.timeoutHandle=Vs(Ln.bind(null,e,He,qt),r);break}Ln(e,He,qt);break;case 5:Ln(e,He,qt);break;default:throw Error(A(329))}}}return Ke(e,ke()),e.callbackNode===n?nm.bind(null,e):null}function cl(e,t){var n=ni;return e.current.memoizedState.isDehydrated&&(On(e,t).flags|=256),e=Ka(e,t),e!==2&&(t=He,He=n,t!==null&&dl(t)),e}function dl(e){He===null?He=e:He.push.apply(He,e)}function qy(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Pt(a(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function rn(e,t){for(t&=~du,t&=~ho,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-zt(t),r=1<<n;e[n]=-1,t&=~r}}function Vc(e){if(ne&6)throw Error(A(327));fr();var t=La(e,0);if(!(t&1))return Ke(e,ke()),null;var n=Ka(e,t);if(e.tag!==0&&n===2){var r=Os(e);r!==0&&(t=r,n=cl(e,r))}if(n===1)throw n=Si,On(e,0),rn(e,t),Ke(e,ke()),n;if(n===6)throw Error(A(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Ln(e,He,qt),Ke(e,ke()),null}function fu(e,t){var n=ne;ne|=1;try{return e(t)}finally{ne=n,ne===0&&(kr=ke()+500,co&&bn())}}function qn(e){on!==null&&on.tag===0&&!(ne&6)&&fr();var t=ne;ne|=1;var n=mt.transition,r=oe;try{if(mt.transition=null,oe=1,e)return e()}finally{oe=r,mt.transition=n,ne=t,!(ne&6)&&bn()}}function mu(){et=or.current,fe(or)}function On(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,xy(n)),Se!==null)for(n=Se.return;n!==null;){var r=n;switch(Xl(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&Oa();break;case 3:vr(),fe(Qe),fe(Me),ru();break;case 5:nu(r);break;case 4:vr();break;case 13:fe(he);break;case 19:fe(he);break;case 10:Zl(r.type._context);break;case 22:case 23:mu()}n=n.return}if(Ne=e,Se=e=hn(e.current,null),Le=et=t,Ee=0,Si=null,du=ho=Bn=0,He=ni=null,In!==null){for(t=0;t<In.length;t++)if(n=In[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,a=n.pending;if(a!==null){var o=a.next;a.next=i,r.next=o}n.pending=r}In=null}return e}function rm(e,t){do{var n=Se;try{if(Gl(),ya.current=Ha,Va){for(var r=ge.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}Va=!1}if(Wn=0,Pe=ze=ge=null,ei=!1,wi=0,cu.current=null,n===null||n.return===null){Ee=1,Si=t,Se=null;break}e:{var a=e,o=n.return,s=n,l=t;if(t=Le,s.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,c=s,d=c.tag;if(!(c.mode&1)&&(d===0||d===11||d===15)){var p=c.alternate;p?(c.updateQueue=p.updateQueue,c.memoizedState=p.memoizedState,c.lanes=p.lanes):(c.updateQueue=null,c.memoizedState=null)}var f=Lc(o);if(f!==null){f.flags&=-257,Ac(f,o,s,a,t),f.mode&1&&Tc(a,u,t),t=f,l=u;var w=t.updateQueue;if(w===null){var _=new Set;_.add(l),t.updateQueue=_}else w.add(l);break e}else{if(!(t&1)){Tc(a,u,t),hu();break e}l=Error(A(426))}}else if(me&&s.mode&1){var S=Lc(o);if(S!==null){!(S.flags&65536)&&(S.flags|=256),Ac(S,o,s,a,t),Kl(wr(l,s));break e}}a=l=wr(l,s),Ee!==4&&(Ee=2),ni===null?ni=[a]:ni.push(a),a=o;do{switch(a.tag){case 3:a.flags|=65536,t&=-t,a.lanes|=t;var h=Wf(a,l,t);Sc(a,h);break e;case 1:s=l;var m=a.type,g=a.stateNode;if(!(a.flags&128)&&(typeof m.getDerivedStateFromError=="function"||g!==null&&typeof g.componentDidCatch=="function"&&(fn===null||!fn.has(g)))){a.flags|=65536,t&=-t,a.lanes|=t;var b=Bf(a,s,t);Sc(a,b);break e}}a=a.return}while(a!==null)}om(n)}catch(z){t=z,Se===n&&n!==null&&(Se=n=n.return);continue}break}while(!0)}function im(){var e=$a.current;return $a.current=Ha,e===null?Ha:e}function hu(){(Ee===0||Ee===3||Ee===2)&&(Ee=4),Ne===null||!(Bn&268435455)&&!(ho&268435455)||rn(Ne,Le)}function Ka(e,t){var n=ne;ne|=2;var r=im();(Ne!==e||Le!==t)&&(qt=null,On(e,t));do try{Uy();break}catch(i){rm(e,i)}while(!0);if(Gl(),ne=n,$a.current=r,Se!==null)throw Error(A(261));return Ne=null,Le=0,Ee}function Uy(){for(;Se!==null;)am(Se)}function Vy(){for(;Se!==null&&!hg();)am(Se)}function am(e){var t=lm(e.alternate,e,et);e.memoizedProps=e.pendingProps,t===null?om(e):Se=t,cu.current=null}function om(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=My(n,t),n!==null){n.flags&=32767,Se=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Ee=6,Se=null;return}}else if(n=Oy(n,t,et),n!==null){Se=n;return}if(t=t.sibling,t!==null){Se=t;return}Se=t=e}while(t!==null);Ee===0&&(Ee=5)}function Ln(e,t,n){var r=oe,i=mt.transition;try{mt.transition=null,oe=1,Hy(e,t,n,r)}finally{mt.transition=i,oe=r}return null}function Hy(e,t,n,r){do fr();while(on!==null);if(ne&6)throw Error(A(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(A(177));e.callbackNode=null,e.callbackPriority=0;var a=n.lanes|n.childLanes;if(Cg(e,a),e===Ne&&(Se=Ne=null,Le=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||na||(na=!0,um(Ta,function(){return fr(),null})),a=(n.flags&15990)!==0,n.subtreeFlags&15990||a){a=mt.transition,mt.transition=null;var o=oe;oe=1;var s=ne;ne|=4,cu.current=null,Fy(e,n),em(n,e),dy(qs),Aa=!!Bs,qs=Bs=null,e.current=n,Wy(n),gg(),ne=s,oe=o,mt.transition=a}else e.current=n;if(na&&(na=!1,on=e,Xa=i),a=e.pendingLanes,a===0&&(fn=null),_g(n.stateNode),Ke(e,ke()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(Qa)throw Qa=!1,e=ll,ll=null,e;return Xa&1&&e.tag!==0&&fr(),a=e.pendingLanes,a&1?e===ul?ri++:(ri=0,ul=e):ri=0,bn(),null}function fr(){if(on!==null){var e=Fp(Xa),t=mt.transition,n=oe;try{if(mt.transition=null,oe=16>e?16:e,on===null)var r=!1;else{if(e=on,on=null,Xa=0,ne&6)throw Error(A(331));var i=ne;for(ne|=4,q=e.current;q!==null;){var a=q,o=a.child;if(q.flags&16){var s=a.deletions;if(s!==null){for(var l=0;l<s.length;l++){var u=s[l];for(q=u;q!==null;){var c=q;switch(c.tag){case 0:case 11:case 15:ti(8,c,a)}var d=c.child;if(d!==null)d.return=c,q=d;else for(;q!==null;){c=q;var p=c.sibling,f=c.return;if(Gf(c),c===u){q=null;break}if(p!==null){p.return=f,q=p;break}q=f}}}var w=a.alternate;if(w!==null){var _=w.child;if(_!==null){w.child=null;do{var S=_.sibling;_.sibling=null,_=S}while(_!==null)}}q=a}}if(a.subtreeFlags&2064&&o!==null)o.return=a,q=o;else e:for(;q!==null;){if(a=q,a.flags&2048)switch(a.tag){case 0:case 11:case 15:ti(9,a,a.return)}var h=a.sibling;if(h!==null){h.return=a.return,q=h;break e}q=a.return}}var m=e.current;for(q=m;q!==null;){o=q;var g=o.child;if(o.subtreeFlags&2064&&g!==null)g.return=o,q=g;else e:for(o=m;q!==null;){if(s=q,s.flags&2048)try{switch(s.tag){case 0:case 11:case 15:mo(9,s)}}catch(z){_e(s,s.return,z)}if(s===o){q=null;break e}var b=s.sibling;if(b!==null){b.return=s.return,q=b;break e}q=s.return}}if(ne=i,bn(),Ot&&typeof Ot.onPostCommitFiberRoot=="function")try{Ot.onPostCommitFiberRoot(ao,e)}catch{}r=!0}return r}finally{oe=n,mt.transition=t}}return!1}function Hc(e,t,n){t=wr(n,t),t=Wf(e,t,1),e=pn(e,t,1),t=Be(),e!==null&&(Li(e,1,t),Ke(e,t))}function _e(e,t,n){if(e.tag===3)Hc(e,e,n);else for(;t!==null;){if(t.tag===3){Hc(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(fn===null||!fn.has(r))){e=wr(n,e),e=Bf(t,e,1),t=pn(t,e,1),e=Be(),t!==null&&(Li(t,1,e),Ke(t,e));break}}t=t.return}}function $y(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=Be(),e.pingedLanes|=e.suspendedLanes&n,Ne===e&&(Le&n)===n&&(Ee===4||Ee===3&&(Le&130023424)===Le&&500>ke()-pu?On(e,0):du|=n),Ke(e,t)}function sm(e,t){t===0&&(e.mode&1?(t=$i,$i<<=1,!($i&130023424)&&($i=4194304)):t=1);var n=Be();e=Kt(e,t),e!==null&&(Li(e,t,n),Ke(e,n))}function Qy(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),sm(e,n)}function Xy(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(A(314))}r!==null&&r.delete(t),sm(e,n)}var lm;lm=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Qe.current)$e=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return $e=!1,jy(e,t,n);$e=!!(e.flags&131072)}else $e=!1,me&&t.flags&1048576&&pf(t,Fa,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;_a(e,t),e=t.pendingProps;var i=yr(t,Me.current);pr(t,n),i=au(null,t,r,e,i,n);var a=ou();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,Xe(r)?(a=!0,Ma(t)):a=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,eu(t),i.updater=fo,t.stateNode=i,i._reactInternals=t,Gs(t,r,e,n),t=el(null,t,r,!0,a,n)):(t.tag=0,me&&a&&Ql(t),We(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(_a(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=Yy(r),e=bt(r,e),i){case 0:t=Js(null,t,r,e,n);break e;case 1:t=jc(null,t,r,e,n);break e;case 11:t=Rc(null,t,r,e,n);break e;case 14:t=Ic(null,t,r,bt(r.type,e),n);break e}throw Error(A(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:bt(r,i),Js(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:bt(r,i),jc(e,t,r,i,n);case 3:e:{if(Hf(t),e===null)throw Error(A(387));r=t.pendingProps,a=t.memoizedState,i=a.element,xf(e,t),qa(t,r,null,n);var o=t.memoizedState;if(r=o.element,a.isDehydrated)if(a={element:r,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){i=wr(Error(A(423)),t),t=Oc(e,t,r,n,i);break e}else if(r!==i){i=wr(Error(A(424)),t),t=Oc(e,t,r,n,i);break e}else for(nt=dn(t.stateNode.containerInfo.firstChild),it=t,me=!0,Ct=null,n=gf(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(xr(),r===i){t=Yt(e,t,n);break e}We(e,t,r,n)}t=t.child}return t;case 5:return _f(t),e===null&&Xs(t),r=t.type,i=t.pendingProps,a=e!==null?e.memoizedProps:null,o=i.children,Us(r,i)?o=null:a!==null&&Us(r,a)&&(t.flags|=32),Vf(e,t),We(e,t,o,n),t.child;case 6:return e===null&&Xs(t),null;case 13:return $f(e,t,n);case 4:return tu(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=_r(t,null,r,n):We(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:bt(r,i),Rc(e,t,r,i,n);case 7:return We(e,t,t.pendingProps,n),t.child;case 8:return We(e,t,t.pendingProps.children,n),t.child;case 12:return We(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,a=t.memoizedProps,o=i.value,ce(Wa,r._currentValue),r._currentValue=o,a!==null)if(Pt(a.value,o)){if(a.children===i.children&&!Qe.current){t=Yt(e,t,n);break e}}else for(a=t.child,a!==null&&(a.return=t);a!==null;){var s=a.dependencies;if(s!==null){o=a.child;for(var l=s.firstContext;l!==null;){if(l.context===r){if(a.tag===1){l=$t(-1,n&-n),l.tag=2;var u=a.updateQueue;if(u!==null){u=u.shared;var c=u.pending;c===null?l.next=l:(l.next=c.next,c.next=l),u.pending=l}}a.lanes|=n,l=a.alternate,l!==null&&(l.lanes|=n),Ks(a.return,n,t),s.lanes|=n;break}l=l.next}}else if(a.tag===10)o=a.type===t.type?null:a.child;else if(a.tag===18){if(o=a.return,o===null)throw Error(A(341));o.lanes|=n,s=o.alternate,s!==null&&(s.lanes|=n),Ks(o,n,t),o=a.sibling}else o=a.child;if(o!==null)o.return=a;else for(o=a;o!==null;){if(o===t){o=null;break}if(a=o.sibling,a!==null){a.return=o.return,o=a;break}o=o.return}a=o}We(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,pr(t,n),i=ht(i),r=r(i),t.flags|=1,We(e,t,r,n),t.child;case 14:return r=t.type,i=bt(r,t.pendingProps),i=bt(r.type,i),Ic(e,t,r,i,n);case 15:return qf(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:bt(r,i),_a(e,t),t.tag=1,Xe(r)?(e=!0,Ma(t)):e=!1,pr(t,n),Ff(t,r,i),Gs(t,r,i,n),el(null,t,r,!0,e,n);case 19:return Qf(e,t,n);case 22:return Uf(e,t,n)}throw Error(A(156,t.tag))};function um(e,t){return jp(e,t)}function Ky(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function ft(e,t,n,r){return new Ky(e,t,n,r)}function gu(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Yy(e){if(typeof e=="function")return gu(e)?1:0;if(e!=null){if(e=e.$$typeof,e===jl)return 11;if(e===Ol)return 14}return 2}function hn(e,t){var n=e.alternate;return n===null?(n=ft(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function ka(e,t,n,r,i,a){var o=2;if(r=e,typeof e=="function")gu(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case Yn:return Mn(n.children,i,a,t);case Il:o=8,i|=8;break;case ws:return e=ft(12,n,t,i|2),e.elementType=ws,e.lanes=a,e;case ks:return e=ft(13,n,t,i),e.elementType=ks,e.lanes=a,e;case bs:return e=ft(19,n,t,i),e.elementType=bs,e.lanes=a,e;case xp:return go(n,i,a,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case gp:o=10;break e;case yp:o=9;break e;case jl:o=11;break e;case Ol:o=14;break e;case en:o=16,r=null;break e}throw Error(A(130,e==null?e:typeof e,""))}return t=ft(o,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function Mn(e,t,n,r){return e=ft(7,e,r,t),e.lanes=n,e}function go(e,t,n,r){return e=ft(22,e,r,t),e.elementType=xp,e.lanes=n,e.stateNode={isHidden:!1},e}function Jo(e,t,n){return e=ft(6,e,null,t),e.lanes=n,e}function es(e,t,n){return t=ft(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Gy(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Io(0),this.expirationTimes=Io(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Io(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function yu(e,t,n,r,i,a,o,s,l){return e=new Gy(e,t,n,s,l),t===1?(t=1,a===!0&&(t|=8)):t=0,a=ft(3,null,null,t),e.current=a,a.stateNode=e,a.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},eu(a),e}function Zy(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:Kn,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function cm(e){if(!e)return vn;e=e._reactInternals;e:{if(Vn(e)!==e||e.tag!==1)throw Error(A(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(Xe(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(A(171))}if(e.tag===1){var n=e.type;if(Xe(n))return cf(e,n,t)}return t}function dm(e,t,n,r,i,a,o,s,l){return e=yu(n,r,!0,e,i,a,o,s,l),e.context=cm(null),n=e.current,r=Be(),i=mn(n),a=$t(r,i),a.callback=t??null,pn(n,a,i),e.current.lanes=i,Li(e,i,r),Ke(e,r),e}function yo(e,t,n,r){var i=t.current,a=Be(),o=mn(i);return n=cm(n),t.context===null?t.context=n:t.pendingContext=n,t=$t(a,o),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=pn(i,t,o),e!==null&&(Et(e,i,o,a),ga(e,i,o)),o}function Ya(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function $c(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function xu(e,t){$c(e,t),(e=e.alternate)&&$c(e,t)}function Jy(){return null}var pm=typeof reportError=="function"?reportError:function(e){console.error(e)};function _u(e){this._internalRoot=e}xo.prototype.render=_u.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(A(409));yo(e,t,null,null)};xo.prototype.unmount=_u.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;qn(function(){yo(null,e,null,null)}),t[Xt]=null}};function xo(e){this._internalRoot=e}xo.prototype.unstable_scheduleHydration=function(e){if(e){var t=qp();e={blockedOn:null,target:e,priority:t};for(var n=0;n<nn.length&&t!==0&&t<nn[n].priority;n++);nn.splice(n,0,e),n===0&&Vp(e)}};function vu(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function _o(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function Qc(){}function ex(e,t,n,r,i){if(i){if(typeof r=="function"){var a=r;r=function(){var u=Ya(o);a.call(u)}}var o=dm(t,r,e,0,null,!1,!1,"",Qc);return e._reactRootContainer=o,e[Xt]=o.current,gi(e.nodeType===8?e.parentNode:e),qn(),o}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var s=r;r=function(){var u=Ya(l);s.call(u)}}var l=yu(e,0,!1,null,null,!1,!1,"",Qc);return e._reactRootContainer=l,e[Xt]=l.current,gi(e.nodeType===8?e.parentNode:e),qn(function(){yo(t,l,n,r)}),l}function vo(e,t,n,r,i){var a=n._reactRootContainer;if(a){var o=a;if(typeof i=="function"){var s=i;i=function(){var l=Ya(o);s.call(l)}}yo(t,o,e,i)}else o=ex(n,t,e,i,r);return Ya(o)}Wp=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=$r(t.pendingLanes);n!==0&&(Fl(t,n|1),Ke(t,ke()),!(ne&6)&&(kr=ke()+500,bn()))}break;case 13:qn(function(){var r=Kt(e,1);if(r!==null){var i=Be();Et(r,e,1,i)}}),xu(e,1)}};Wl=function(e){if(e.tag===13){var t=Kt(e,134217728);if(t!==null){var n=Be();Et(t,e,134217728,n)}xu(e,134217728)}};Bp=function(e){if(e.tag===13){var t=mn(e),n=Kt(e,t);if(n!==null){var r=Be();Et(n,e,t,r)}xu(e,t)}};qp=function(){return oe};Up=function(e,t){var n=oe;try{return oe=e,t()}finally{oe=n}};Rs=function(e,t,n){switch(t){case"input":if(zs(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=uo(r);if(!i)throw Error(A(90));vp(r),zs(r,i)}}}break;case"textarea":kp(e,n);break;case"select":t=n.value,t!=null&&lr(e,!!n.multiple,t,!1)}};Np=fu;Tp=qn;var tx={usingClientEntryPoint:!1,Events:[Ri,er,uo,Ep,Pp,fu]},Wr={findFiberByHostInstance:Rn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},nx={bundleType:Wr.bundleType,version:Wr.version,rendererPackageName:Wr.rendererPackageName,rendererConfig:Wr.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:Gt.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=Rp(e),e===null?null:e.stateNode},findFiberByHostInstance:Wr.findFiberByHostInstance||Jy,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var ra=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!ra.isDisabled&&ra.supportsFiber)try{ao=ra.inject(nx),Ot=ra}catch{}}ot.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=tx;ot.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!vu(t))throw Error(A(200));return Zy(e,t,null,n)};ot.createRoot=function(e,t){if(!vu(e))throw Error(A(299));var n=!1,r="",i=pm;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=yu(e,1,!1,null,null,n,!1,r,i),e[Xt]=t.current,gi(e.nodeType===8?e.parentNode:e),new _u(t)};ot.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(A(188)):(e=Object.keys(e).join(","),Error(A(268,e)));return e=Rp(t),e=e===null?null:e.stateNode,e};ot.flushSync=function(e){return qn(e)};ot.hydrate=function(e,t,n){if(!_o(t))throw Error(A(200));return vo(null,e,t,!0,n)};ot.hydrateRoot=function(e,t,n){if(!vu(e))throw Error(A(405));var r=n!=null&&n.hydratedSources||null,i=!1,a="",o=pm;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(a=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=dm(t,null,e,1,n??null,i,!1,a,o),e[Xt]=t.current,gi(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new xo(t)};ot.render=function(e,t,n){if(!_o(t))throw Error(A(200));return vo(null,e,t,!1,n)};ot.unmountComponentAtNode=function(e){if(!_o(e))throw Error(A(40));return e._reactRootContainer?(qn(function(){vo(null,null,e,!1,function(){e._reactRootContainer=null,e[Xt]=null})}),!0):!1};ot.unstable_batchedUpdates=fu;ot.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!_o(n))throw Error(A(200));if(e==null||e._reactInternals===void 0)throw Error(A(38));return vo(e,t,n,!1,r)};ot.version="18.3.1-next-f1338f8080-20240426";function fm(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(fm)}catch(e){console.error(e)}}fm(),pp.exports=ot;var rx=pp.exports,Xc=rx;_s.createRoot=Xc.createRoot,_s.hydrateRoot=Xc.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ci(){return Ci=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ci.apply(this,arguments)}var sn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(sn||(sn={}));const Kc="popstate";function ix(e){e===void 0&&(e={});function t(r,i){let{pathname:a,search:o,hash:s}=r.location;return pl("",{pathname:a,search:o,hash:s},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:Ga(i)}return ox(t,n,null,e)}function ve(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function wu(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function ax(){return Math.random().toString(36).substr(2,8)}function Yc(e,t){return{usr:e.state,key:e.key,idx:t}}function pl(e,t,n,r){return n===void 0&&(n=null),Ci({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?Pr(t):t,{state:n,key:t&&t.key||r||ax()})}function Ga(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function Pr(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function ox(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:a=!1}=r,o=i.history,s=sn.Pop,l=null,u=c();u==null&&(u=0,o.replaceState(Ci({},o.state,{idx:u}),""));function c(){return(o.state||{idx:null}).idx}function d(){s=sn.Pop;let S=c(),h=S==null?null:S-u;u=S,l&&l({action:s,location:_.location,delta:h})}function p(S,h){s=sn.Push;let m=pl(_.location,S,h);u=c()+1;let g=Yc(m,u),b=_.createHref(m);try{o.pushState(g,"",b)}catch(z){if(z instanceof DOMException&&z.name==="DataCloneError")throw z;i.location.assign(b)}a&&l&&l({action:s,location:_.location,delta:1})}function f(S,h){s=sn.Replace;let m=pl(_.location,S,h);u=c();let g=Yc(m,u),b=_.createHref(m);o.replaceState(g,"",b),a&&l&&l({action:s,location:_.location,delta:0})}function w(S){let h=i.location.origin!=="null"?i.location.origin:i.location.href,m=typeof S=="string"?S:Ga(S);return m=m.replace(/ $/,"%20"),ve(h,"No window.location.(origin|href) available to create URL for href: "+m),new URL(m,h)}let _={get action(){return s},get location(){return e(i,o)},listen(S){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(Kc,d),l=S,()=>{i.removeEventListener(Kc,d),l=null}},createHref(S){return t(i,S)},createURL:w,encodeLocation(S){let h=w(S);return{pathname:h.pathname,search:h.search,hash:h.hash}},push:p,replace:f,go(S){return o.go(S)}};return _}var Gc;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(Gc||(Gc={}));function sx(e,t,n){return n===void 0&&(n="/"),lx(e,t,n)}function lx(e,t,n,r){let i=typeof t=="string"?Pr(t):t,a=br(i.pathname||"/",n);if(a==null)return null;let o=mm(e);ux(o);let s=null;for(let l=0;s==null&&l<o.length;++l){let u=vx(a);s=xx(o[l],u)}return s}function mm(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(a,o,s)=>{let l={relativePath:s===void 0?a.path||"":s,caseSensitive:a.caseSensitive===!0,childrenIndex:o,route:a};l.relativePath.startsWith("/")&&(ve(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=gn([r,l.relativePath]),c=n.concat(l);a.children&&a.children.length>0&&(ve(a.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),mm(a.children,t,c,u)),!(a.path==null&&!a.index)&&t.push({path:u,score:gx(u,a.index),routesMeta:c})};return e.forEach((a,o)=>{var s;if(a.path===""||!((s=a.path)!=null&&s.includes("?")))i(a,o);else for(let l of hm(a.path))i(a,o,l)}),t}function hm(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),a=n.replace(/\?$/,"");if(r.length===0)return i?[a,""]:[a];let o=hm(r.join("/")),s=[];return s.push(...o.map(l=>l===""?a:[a,l].join("/"))),i&&s.push(...o),s.map(l=>e.startsWith("/")&&l===""?"/":l)}function ux(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:yx(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const cx=/^:[\w-]+$/,dx=3,px=2,fx=1,mx=10,hx=-2,Zc=e=>e==="*";function gx(e,t){let n=e.split("/"),r=n.length;return n.some(Zc)&&(r+=hx),t&&(r+=px),n.filter(i=>!Zc(i)).reduce((i,a)=>i+(cx.test(a)?dx:a===""?fx:mx),r)}function yx(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function xx(e,t,n){let{routesMeta:r}=e,i={},a="/",o=[];for(let s=0;s<r.length;++s){let l=r[s],u=s===r.length-1,c=a==="/"?t:t.slice(a.length)||"/",d=fl({path:l.relativePath,caseSensitive:l.caseSensitive,end:u},c),p=l.route;if(!d)return null;Object.assign(i,d.params),o.push({params:i,pathname:gn([a,d.pathname]),pathnameBase:Cx(gn([a,d.pathnameBase])),route:p}),d.pathnameBase!=="/"&&(a=gn([a,d.pathnameBase]))}return o}function fl(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=_x(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let a=i[0],o=a.replace(/(.)\/+$/,"$1"),s=i.slice(1);return{params:r.reduce((u,c,d)=>{let{paramName:p,isOptional:f}=c;if(p==="*"){let _=s[d]||"";o=a.slice(0,a.length-_.length).replace(/(.)\/+$/,"$1")}const w=s[d];return f&&!w?u[p]=void 0:u[p]=(w||"").replace(/%2F/g,"/"),u},{}),pathname:a,pathnameBase:o,pattern:e}}function _x(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),wu(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(o,s,l)=>(r.push({paramName:s,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function vx(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return wu(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function br(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const wx=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,kx=e=>wx.test(e);function bx(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?Pr(e):e,a;if(n)if(kx(n))a=n;else{if(n.includes("//")){let o=n;n=n.replace(/\/\/+/g,"/"),wu(!1,"Pathnames cannot have embedded double slashes - normalizing "+(o+" -> "+n))}n.startsWith("/")?a=Jc(n.substring(1),"/"):a=Jc(n,t)}else a=t;return{pathname:a,search:zx(r),hash:Ex(i)}}function Jc(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function ts(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function Sx(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function gm(e,t){let n=Sx(e);return t?n.map((r,i)=>i===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function ym(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=Pr(e):(i=Ci({},e),ve(!i.pathname||!i.pathname.includes("?"),ts("?","pathname","search",i)),ve(!i.pathname||!i.pathname.includes("#"),ts("#","pathname","hash",i)),ve(!i.search||!i.search.includes("#"),ts("#","search","hash",i)));let a=e===""||i.pathname==="",o=a?"/":i.pathname,s;if(o==null)s=n;else{let d=t.length-1;if(!r&&o.startsWith("..")){let p=o.split("/");for(;p[0]==="..";)p.shift(),d-=1;i.pathname=p.join("/")}s=d>=0?t[d]:"/"}let l=bx(i,s),u=o&&o!=="/"&&o.endsWith("/"),c=(a||o===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||c)&&(l.pathname+="/"),l}const gn=e=>e.join("/").replace(/\/\/+/g,"/"),Cx=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),zx=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,Ex=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function Px(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const xm=["post","put","patch","delete"];new Set(xm);const Nx=["get",...xm];new Set(Nx);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function zi(){return zi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},zi.apply(this,arguments)}const wo=C.createContext(null),_m=C.createContext(null),Sn=C.createContext(null),ko=C.createContext(null),Cn=C.createContext({outlet:null,matches:[],isDataRoute:!1}),vm=C.createContext(null);function Tx(e,t){let{relative:n}=t===void 0?{}:t;ji()||ve(!1);let{basename:r,navigator:i}=C.useContext(Sn),{hash:a,pathname:o,search:s}=bo(e,{relative:n}),l=o;return r!=="/"&&(l=o==="/"?r:gn([r,o])),i.createHref({pathname:l,search:s,hash:a})}function ji(){return C.useContext(ko)!=null}function Oi(){return ji()||ve(!1),C.useContext(ko).location}function wm(e){C.useContext(Sn).static||C.useLayoutEffect(e)}function Lx(){let{isDataRoute:e}=C.useContext(Cn);return e?Vx():Ax()}function Ax(){ji()||ve(!1);let e=C.useContext(wo),{basename:t,future:n,navigator:r}=C.useContext(Sn),{matches:i}=C.useContext(Cn),{pathname:a}=Oi(),o=JSON.stringify(gm(i,n.v7_relativeSplatPath)),s=C.useRef(!1);return wm(()=>{s.current=!0}),C.useCallback(function(u,c){if(c===void 0&&(c={}),!s.current)return;if(typeof u=="number"){r.go(u);return}let d=ym(u,JSON.parse(o),a,c.relative==="path");e==null&&t!=="/"&&(d.pathname=d.pathname==="/"?t:gn([t,d.pathname])),(c.replace?r.replace:r.push)(d,c.state,c)},[t,r,o,a,e])}function km(){let{matches:e}=C.useContext(Cn),t=e[e.length-1];return t?t.params:{}}function bo(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=C.useContext(Sn),{matches:i}=C.useContext(Cn),{pathname:a}=Oi(),o=JSON.stringify(gm(i,r.v7_relativeSplatPath));return C.useMemo(()=>ym(e,JSON.parse(o),a,n==="path"),[e,o,a,n])}function Rx(e,t){return Ix(e,t)}function Ix(e,t,n,r){ji()||ve(!1);let{navigator:i}=C.useContext(Sn),{matches:a}=C.useContext(Cn),o=a[a.length-1],s=o?o.params:{};o&&o.pathname;let l=o?o.pathnameBase:"/";o&&o.route;let u=Oi(),c;if(t){var d;let S=typeof t=="string"?Pr(t):t;l==="/"||(d=S.pathname)!=null&&d.startsWith(l)||ve(!1),c=S}else c=u;let p=c.pathname||"/",f=p;if(l!=="/"){let S=l.replace(/^\//,"").split("/");f="/"+p.replace(/^\//,"").split("/").slice(S.length).join("/")}let w=sx(e,{pathname:f}),_=Fx(w&&w.map(S=>Object.assign({},S,{params:Object.assign({},s,S.params),pathname:gn([l,i.encodeLocation?i.encodeLocation(S.pathname).pathname:S.pathname]),pathnameBase:S.pathnameBase==="/"?l:gn([l,i.encodeLocation?i.encodeLocation(S.pathnameBase).pathname:S.pathnameBase])})),a,n,r);return t&&_?C.createElement(ko.Provider,{value:{location:zi({pathname:"/",search:"",hash:"",state:null,key:"default"},c),navigationType:sn.Pop}},_):_}function jx(){let e=Ux(),t=Px(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return C.createElement(C.Fragment,null,C.createElement("h2",null,"Unexpected Application Error!"),C.createElement("h3",{style:{fontStyle:"italic"}},t),n?C.createElement("pre",{style:i},n):null,null)}const Ox=C.createElement(jx,null);class Mx extends C.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?C.createElement(Cn.Provider,{value:this.props.routeContext},C.createElement(vm.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function Dx(e){let{routeContext:t,match:n,children:r}=e,i=C.useContext(wo);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),C.createElement(Cn.Provider,{value:t},r)}function Fx(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var a;if(!n)return null;if(n.errors)e=n.matches;else if((a=r)!=null&&a.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let o=e,s=(i=n)==null?void 0:i.errors;if(s!=null){let c=o.findIndex(d=>d.route.id&&(s==null?void 0:s[d.route.id])!==void 0);c>=0||ve(!1),o=o.slice(0,Math.min(o.length,c+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let c=0;c<o.length;c++){let d=o[c];if((d.route.HydrateFallback||d.route.hydrateFallbackElement)&&(u=c),d.route.id){let{loaderData:p,errors:f}=n,w=d.route.loader&&p[d.route.id]===void 0&&(!f||f[d.route.id]===void 0);if(d.route.lazy||w){l=!0,u>=0?o=o.slice(0,u+1):o=[o[0]];break}}}return o.reduceRight((c,d,p)=>{let f,w=!1,_=null,S=null;n&&(f=s&&d.route.id?s[d.route.id]:void 0,_=d.route.errorElement||Ox,l&&(u<0&&p===0?(Hx("route-fallback"),w=!0,S=null):u===p&&(w=!0,S=d.route.hydrateFallbackElement||null)));let h=t.concat(o.slice(0,p+1)),m=()=>{let g;return f?g=_:w?g=S:d.route.Component?g=C.createElement(d.route.Component,null):d.route.element?g=d.route.element:g=c,C.createElement(Dx,{match:d,routeContext:{outlet:c,matches:h,isDataRoute:n!=null},children:g})};return n&&(d.route.ErrorBoundary||d.route.errorElement||p===0)?C.createElement(Mx,{location:n.location,revalidation:n.revalidation,component:_,error:f,children:m(),routeContext:{outlet:null,matches:h,isDataRoute:!0}}):m()},null)}var bm=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(bm||{}),Sm=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Sm||{});function Wx(e){let t=C.useContext(wo);return t||ve(!1),t}function Bx(e){let t=C.useContext(_m);return t||ve(!1),t}function qx(e){let t=C.useContext(Cn);return t||ve(!1),t}function Cm(e){let t=qx(),n=t.matches[t.matches.length-1];return n.route.id||ve(!1),n.route.id}function Ux(){var e;let t=C.useContext(vm),n=Bx(),r=Cm();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function Vx(){let{router:e}=Wx(bm.UseNavigateStable),t=Cm(Sm.UseNavigateStable),n=C.useRef(!1);return wm(()=>{n.current=!0}),C.useCallback(function(i,a){a===void 0&&(a={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,zi({fromRouteId:t},a)))},[e,t])}const ed={};function Hx(e,t,n){ed[e]||(ed[e]=!0)}function $x(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function ba(e){ve(!1)}function Qx(e){let{basename:t="/",children:n=null,location:r,navigationType:i=sn.Pop,navigator:a,static:o=!1,future:s}=e;ji()&&ve(!1);let l=t.replace(/^\/*/,"/"),u=C.useMemo(()=>({basename:l,navigator:a,static:o,future:zi({v7_relativeSplatPath:!1},s)}),[l,s,a,o]);typeof r=="string"&&(r=Pr(r));let{pathname:c="/",search:d="",hash:p="",state:f=null,key:w="default"}=r,_=C.useMemo(()=>{let S=br(c,l);return S==null?null:{location:{pathname:S,search:d,hash:p,state:f,key:w},navigationType:i}},[l,c,d,p,f,w,i]);return _==null?null:C.createElement(Sn.Provider,{value:u},C.createElement(ko.Provider,{children:n,value:_}))}function Xx(e){let{children:t,location:n}=e;return Rx(ml(t),n)}new Promise(()=>{});function ml(e,t){t===void 0&&(t=[]);let n=[];return C.Children.forEach(e,(r,i)=>{if(!C.isValidElement(r))return;let a=[...t,i];if(r.type===C.Fragment){n.push.apply(n,ml(r.props.children,a));return}r.type!==ba&&ve(!1),!r.props.index||!r.props.children||ve(!1);let o={id:r.props.id||a.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(o.children=ml(r.props.children,a)),n.push(o)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Za(){return Za=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Za.apply(this,arguments)}function zm(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,a;for(a=0;a<r.length;a++)i=r[a],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function Kx(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function Yx(e,t){return e.button===0&&(!t||t==="_self")&&!Kx(e)}const Gx=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],Zx=["aria-current","caseSensitive","className","end","style","to","viewTransition","children"],Jx="6";try{window.__reactRouterVersion=Jx}catch{}const e0=C.createContext({isTransitioning:!1}),t0="startTransition",td=$h[t0];function n0(e){let{basename:t,children:n,future:r,window:i}=e,a=C.useRef();a.current==null&&(a.current=ix({window:i,v5Compat:!0}));let o=a.current,[s,l]=C.useState({action:o.action,location:o.location}),{v7_startTransition:u}=r||{},c=C.useCallback(d=>{u&&td?td(()=>l(d)):l(d)},[l,u]);return C.useLayoutEffect(()=>o.listen(c),[o,c]),C.useEffect(()=>$x(r),[r]),C.createElement(Qx,{basename:t,children:n,location:s.location,navigationType:s.action,navigator:o,future:r})}const r0=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",i0=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,yn=C.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:a,replace:o,state:s,target:l,to:u,preventScrollReset:c,viewTransition:d}=t,p=zm(t,Gx),{basename:f}=C.useContext(Sn),w,_=!1;if(typeof u=="string"&&i0.test(u)&&(w=u,r0))try{let g=new URL(window.location.href),b=u.startsWith("//")?new URL(g.protocol+u):new URL(u),z=br(b.pathname,f);b.origin===g.origin&&z!=null?u=z+b.search+b.hash:_=!0}catch{}let S=Tx(u,{relative:i}),h=s0(u,{replace:o,state:s,target:l,preventScrollReset:c,relative:i,viewTransition:d});function m(g){r&&r(g),g.defaultPrevented||h(g)}return C.createElement("a",Za({},p,{href:w||S,onClick:_||a?r:m,ref:n,target:l}))}),a0=C.forwardRef(function(t,n){let{"aria-current":r="page",caseSensitive:i=!1,className:a="",end:o=!1,style:s,to:l,viewTransition:u,children:c}=t,d=zm(t,Zx),p=bo(l,{relative:d.relative}),f=Oi(),w=C.useContext(_m),{navigator:_,basename:S}=C.useContext(Sn),h=w!=null&&l0(p)&&u===!0,m=_.encodeLocation?_.encodeLocation(p).pathname:p.pathname,g=f.pathname,b=w&&w.navigation&&w.navigation.location?w.navigation.location.pathname:null;i||(g=g.toLowerCase(),b=b?b.toLowerCase():null,m=m.toLowerCase()),b&&S&&(b=br(b,S)||b);const z=m!=="/"&&m.endsWith("/")?m.length-1:m.length;let k=g===m||!o&&g.startsWith(m)&&g.charAt(z)==="/",E=b!=null&&(b===m||!o&&b.startsWith(m)&&b.charAt(m.length)==="/"),N={isActive:k,isPending:E,isTransitioning:h},O=k?r:void 0,L;typeof a=="function"?L=a(N):L=[a,k?"active":null,E?"pending":null,h?"transitioning":null].filter(Boolean).join(" ");let F=typeof s=="function"?s(N):s;return C.createElement(yn,Za({},d,{"aria-current":O,className:L,ref:n,style:F,to:l,viewTransition:u}),typeof c=="function"?c(N):c)});var hl;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(hl||(hl={}));var nd;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(nd||(nd={}));function o0(e){let t=C.useContext(wo);return t||ve(!1),t}function s0(e,t){let{target:n,replace:r,state:i,preventScrollReset:a,relative:o,viewTransition:s}=t===void 0?{}:t,l=Lx(),u=Oi(),c=bo(e,{relative:o});return C.useCallback(d=>{if(Yx(d,n)){d.preventDefault();let p=r!==void 0?r:Ga(u)===Ga(c);l(e,{replace:p,state:i,preventScrollReset:a,relative:o,viewTransition:s})}},[u,l,c,r,i,n,e,a,o,s])}function l0(e,t){t===void 0&&(t={});let n=C.useContext(e0);n==null&&ve(!1);let{basename:r}=o0(hl.useViewTransitionState),i=bo(e,{relative:t.relative});if(!n.isTransitioning)return!1;let a=br(n.currentLocation.pathname,r)||n.currentLocation.pathname,o=br(n.nextLocation.pathname,r)||n.nextLocation.pathname;return fl(i.pathname,o)!=null||fl(i.pathname,a)!=null}const ns="ml-interview-progress",Em=C.createContext(null);function u0({children:e}){const[t,n]=C.useState(()=>{const u=localStorage.getItem(ns);return u?JSON.parse(u):{}});C.useEffect(()=>{localStorage.setItem(ns,JSON.stringify(t))},[t]);const r=C.useCallback((u,c,d)=>{n(p=>{var f;return{...p,[u]:{...p[u],[c]:{...(f=p[u])==null?void 0:f[c],status:d,lastAttempt:new Date().toISOString()}}}})},[]),i=C.useCallback((u,c,d)=>{n(p=>{var f,w,_;return{...p,[u]:{...p[u],[c]:{...(f=p[u])==null?void 0:f[c],status:((_=(w=p[u])==null?void 0:w[c])==null?void 0:_.status)||"in_progress",code:d,lastAttempt:new Date().toISOString()}}}})},[]),a=C.useCallback((u,c)=>{var p;const d=(p=t[u])==null?void 0:p[c];return{status:(d==null?void 0:d.status)||"not_started",code:d==null?void 0:d.code}},[t]),o=C.useCallback((u,c)=>{const d=t[u];if(!d||c===0)return 0;const p=Object.values(d).filter(f=>f.status==="completed").length;return Math.round(p/c*100)},[t]),s=C.useCallback(u=>{const c=u.reduce((p,f)=>p+f.problemCount,0);if(c===0)return 0;let d=0;for(const p of u){const f=t[p.id];f&&(d+=Object.values(f).filter(w=>w.status==="completed").length)}return Math.round(d/c*100)},[t]),l=C.useCallback(()=>{n({}),localStorage.removeItem(ns)},[]);return x.jsx(Em.Provider,{value:{progress:t,updateProblemStatus:r,saveProblemCode:i,getProblemProgress:a,getSectionProgress:o,getOverallProgress:s,resetProgress:l},children:e})}function Mi(){const e=C.useContext(Em);if(!e)throw new Error("useProgress must be used within a ProgressProvider");return e}const Sr=[{id:"numpy-fundamentals",title:"NumPy Fundamentals",description:"Master NumPy array operations essential for ML development.",icon:"🔢",introduction:"\n# NumPy Fundamentals\n\nNumPy is the backbone of scientific computing in Python. Every ML library builds on it.\n\n## Key Concepts\n\n### Array Creation\n- `np.array()`: Create from lists\n- `np.zeros()`, `np.ones()`: Initialize with values\n- `np.arange()`, `np.linspace()`: Numeric ranges\n- `np.eye()`: Identity matrix\n\n### Indexing & Slicing\n- Basic: `arr[0]`, `arr[1:5]`\n- Advanced: `arr[[0, 2, 4]]`, `arr[arr > 0]`\n- Multi-dimensional: `arr[1, :]`, `arr[:, 2:4]`\n\n### Broadcasting\nArrays of different shapes can be combined:\n- (3, 4) + (4,) → broadcasts to (3, 4)\n- (3, 1) * (1, 4) → broadcasts to (3, 4)\n\n### Aggregations\n- `sum()`, `mean()`, `std()` - with axis parameter\n- `argmax()`, `argmin()` - find indices\n\n### Shape Manipulation\n- `reshape()`: Change dimensions\n- `transpose()`, `.T`: Swap axes\n- `flatten()`, `ravel()`: To 1D\n\nMaster these operations to write efficient ML code!\n    ",problems:["numpy-array-creation","numpy-indexing","numpy-broadcasting","numpy-aggregations","numpy-reshape-transpose"]},{id:"python-basics",title:"Python Basics for ML",description:"Master NumPy arrays and essential Python operations for machine learning.",icon:"🐍",introduction:"\n# Python Basics for Machine Learning\n\nNumPy is the foundation of nearly all machine learning in Python. Before diving into complex algorithms, you need to master array operations.\n\n## Key Concepts\n\n### NumPy Arrays\n- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`\n- **Shape manipulation**: `reshape()`, `flatten()`, `transpose()`\n- **Indexing**: Slicing, boolean indexing, fancy indexing\n\n### Essential Operations\n- **Element-wise**: Addition, multiplication, mathematical functions\n- **Aggregations**: `sum()`, `mean()`, `std()`, `min()`, `max()`\n- **Broadcasting**: Automatic expansion of arrays for operations\n\n### Why It Matters\nEvery ML algorithm processes data as arrays. Understanding these operations lets you:\n- Prepare data efficiently\n- Debug model issues\n- Optimize performance\n\nLet's practice with hands-on problems!\n    ",problems:["numpy-array-sum","numpy-matrix-multiply","numpy-broadcast-add"]},{id:"einsum",title:"Einstein Summation (Einsum)",description:"Master einsum for elegant tensor operations.",icon:"∑",introduction:`
# Einstein Summation (Einsum)

Einsum is a powerful notation for expressing tensor operations concisely.

## Why Einsum?
- Express complex operations in one line
- Often faster than chained operations
- Essential for attention mechanisms

## Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

## Common Patterns

### Basic Operations
\`\`\`
'ij->'      Sum all elements
'ij->i'     Sum along axis 1 (row sums)
'ij->j'     Sum along axis 0 (column sums)
'ij->ji'    Transpose
'ii->i'     Extract diagonal
'ii->'      Trace (sum of diagonal)
\`\`\`

### Matrix Operations
\`\`\`
'ik,kj->ij'   Matrix multiplication (A @ B)
'ij,ij->ij'   Element-wise product (A * B)
'ij,ij->'     Frobenius inner product
'i,j->ij'     Outer product
'i,i->'       Dot product
\`\`\`

### Batch Operations
\`\`\`
'bij,bjk->bik'   Batch matrix multiply
'bqd,bkd->bqk'   Attention scores (Q @ K.T)
'bhqk,bhkd->bhqd' Multi-head attention output
\`\`\`

## Key Insight
Letters that appear on both sides are kept.
Letters that disappear are summed over (contracted).

Master einsum to write clean, efficient deep learning code!
    `,problems:["einsum-basics","einsum-matrix-ops","einsum-batch-ops","einsum-advanced","einsum-vs-matmul"]},{id:"pytorch-basics",title:"PyTorch Basics",description:"Learn PyTorch patterns implemented in NumPy.",icon:"🔥",introduction:`
# PyTorch Basics

Learn PyTorch patterns and concepts. Since Pyodide doesn't support PyTorch, we implement these concepts in NumPy.

## Tensor Operations

### Creation
\`\`\`python
# PyTorch           # NumPy equivalent
torch.tensor()      np.array()
torch.zeros()       np.zeros()
torch.randn()       np.random.randn()
\`\`\`

### Operations
\`\`\`python
# PyTorch           # NumPy equivalent
x + y               x + y
x @ y               x @ y
x.sum(dim=1)        x.sum(axis=1)
x.view(2, -1)       x.reshape(2, -1)
\`\`\`

## Autograd Concepts

PyTorch tracks operations for automatic differentiation:
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # dy/dx = 2x = 4
\`\`\`

We implement this manually to understand the math.

## Module Pattern
\`\`\`python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return x @ self.weight + self.bias
\`\`\`

## Loss Functions
- CrossEntropyLoss: Classification
- MSELoss: Regression
- BCELoss: Binary classification

Learn these patterns to understand PyTorch!
    `,problems:["tensor-creation","tensor-operations","autograd-concepts","nn-modules","loss-functions"]},{id:"data-preprocessing",title:"Data Preprocessing",description:"Learn essential data cleaning and transformation techniques.",icon:"🧹",introduction:`
# Data Preprocessing

Real-world data is messy. Before training any model, you need to clean and transform your data.

## Key Concepts

### Handling Missing Data
- **Detection**: \`np.isnan()\`, \`pd.isnull()\`
- **Strategies**: Remove, fill with mean/median/mode, interpolate

### Feature Scaling
- **Normalization**: Scale to [0, 1] range using min-max scaling
- **Standardization**: Transform to zero mean, unit variance (z-score)
- **When to use**: Most algorithms need scaled features

### Encoding Categorical Data
- **One-Hot Encoding**: Convert categories to binary vectors
- **Label Encoding**: Map categories to integers
- **When to use**: Tree-based models handle label encoding; others need one-hot

### Why It Matters
Garbage in, garbage out. Proper preprocessing can:
- Improve model accuracy by 10-30%
- Prevent training failures
- Reduce overfitting

Let's practice these essential skills!
    `,problems:["normalize-features","handle-missing-data","one-hot-encode"]},{id:"supervised-learning",title:"Supervised Learning",description:"Implement core supervised learning algorithms from scratch.",icon:"📊",introduction:`
# Supervised Learning

Supervised learning is when you train a model on labeled data to make predictions.

## Key Concepts

### Linear Regression
- **Goal**: Predict continuous values
- **Formula**: y = wx + b
- **Training**: Minimize mean squared error (MSE)
- **Gradient descent**: Update weights iteratively

### Logistic Regression
- **Goal**: Binary classification
- **Sigmoid function**: Maps outputs to [0, 1] probabilities
- **Loss**: Binary cross-entropy
- **Decision boundary**: Linear separation of classes

### Decision Trees
- **Concept**: Recursive binary splits based on features
- **Splitting criteria**: Gini impurity, Information gain
- **Advantages**: Interpretable, handles non-linear relationships

### The Interview Perspective
Interviewers often ask you to:
- Implement gradient descent from scratch
- Explain the math behind these algorithms
- Discuss trade-offs between algorithms

Let's build these algorithms!
    `,problems:["linear-regression-gd","logistic-regression","logistic-regression-full","binary-cross-entropy","decision-tree-split"]},{id:"unsupervised-learning",title:"Unsupervised Learning",description:"Implement clustering and dimensionality reduction algorithms.",icon:"🔍",introduction:`
# Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

## Key Concepts

### K-Means Clustering
- **Goal**: Group data into K clusters
- **Algorithm**:
  1. Initialize K centroids randomly
  2. Assign points to nearest centroid
  3. Update centroids as cluster means
  4. Repeat until convergence

### Principal Component Analysis (PCA)
- **Goal**: Reduce dimensionality while preserving variance
- **Steps**:
  1. Center the data (subtract mean)
  2. Compute covariance matrix
  3. Find eigenvectors/eigenvalues
  4. Project onto top components

### Why These Matter
- **K-Means**: Customer segmentation, image compression
- **PCA**: Feature reduction, visualization, noise removal

### Interview Tips
- Be ready to implement K-means from scratch
- Explain variance explained by components
- Discuss limitations (K-means assumes spherical clusters)

Let's implement these algorithms!
    `,problems:["kmeans-clustering","pca-implementation"]},{id:"model-evaluation",title:"Model Evaluation",description:"Learn metrics and techniques to evaluate ML models properly.",icon:"📈",introduction:`
# Model Evaluation

Building a model is only half the job. You need to evaluate it properly.

## Key Concepts

### Classification Metrics
- **Accuracy**: Correct / Total (can be misleading!)
- **Precision**: TP / (TP + FP) - "Of positive predictions, how many correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many found?"
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root MSE (same units as target)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained

### Cross-Validation
- **Why**: Single train/test split is unreliable
- **K-Fold**: Split data into K parts, rotate test set
- **Stratified**: Maintain class distribution in folds

### Interview Essentials
- Know when to use each metric
- Implement cross-validation from scratch
- Discuss precision-recall trade-offs

Let's practice evaluation!
    `,problems:["precision-recall-f1","cross-validation","confusion-matrix"]},{id:"deep-learning",title:"Deep Learning Basics",description:"Activation functions and fundamental building blocks.",icon:"⚡",introduction:`
# Deep Learning Basics

Core building blocks that appear in every neural network.

## Key Concepts

### Activation Functions
- **ReLU**: max(0, x) - Most common, avoids vanishing gradient
- **Sigmoid**: 1/(1+e^-x) - Output in [0,1], used for binary classification
- **Softmax**: Converts logits to probabilities that sum to 1
- **Tanh**: Output in [-1, 1], zero-centered

### When to Use Each
- **Hidden layers**: ReLU (or variants like LeakyReLU)
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **RNNs**: Tanh (historically)

### Dense Layers
- **Forward pass**: output = activation(W @ x + b)
- **Parameters**: Weights W and biases b
- **Computation**: Matrix multiplication + bias + activation

Let's implement these fundamentals!
    `,problems:["perceptron","neural-network-forward","backpropagation"]},{id:"neural-networks",title:"Neural Networks",description:"Build neural networks from scratch with forward and backward passes.",icon:"🧠",introduction:`
# Neural Networks

Neural networks are the foundation of deep learning. Understanding the math behind them is essential for ML interviews.

## Key Concepts

### Multi-Layer Perceptron (MLP)
- **Architecture**: Input → Hidden layers → Output
- **Forward pass**: Compute activations layer by layer
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax

### Backpropagation
- **Chain rule**: Compute gradients layer by layer
- **Weight updates**: Gradient descent on all parameters
- **Key insight**: Errors propagate backwards through the network

### Training Techniques
- **Weight Initialization**: Xavier, He initialization
- **Batch Normalization**: Normalize activations for stable training
- **Dropout**: Randomly drop neurons to prevent overfitting

### Loss Functions
- **Cross-entropy**: For classification
- **MSE**: For regression

### Common Interview Topics
- Implement forward/backward pass from scratch
- Explain vanishing/exploding gradients
- Discuss initialization strategies
- Implement regularization techniques

Let's build neural networks from scratch!
    `,problems:["cross-entropy-loss","mlp-forward","backprop-gradients","weight-init","batch-norm","dropout"]},{id:"cnn",title:"Convolutional Neural Networks",description:"Understand convolutions, pooling, and CNN architectures.",icon:"🖼️",introduction:`
# Convolutional Neural Networks (CNNs)

CNNs are the backbone of computer vision. They learn hierarchical features from images.

## Key Concepts

### Convolution Operation
- **Kernel/Filter**: Small matrix that slides over input
- **Feature maps**: Output of applying filters
- **Parameters**: Stride, padding, kernel size

### Pooling Layers
- **Max pooling**: Take maximum in each region
- **Average pooling**: Take average in each region
- **Purpose**: Reduce spatial dimensions, add translation invariance

### CNN Architecture Components
1. **Convolutional layers**: Extract features
2. **Pooling layers**: Reduce dimensions
3. **Flatten**: Convert 2D to 1D
4. **Fully connected**: Classification

### Output Size Formula
\`\`\`
output = (input - kernel + 2*padding) / stride + 1
\`\`\`

### Classic Architectures
- **LeNet**: First successful CNN
- **AlexNet**: Deep CNNs for ImageNet
- **VGG**: Very deep, small kernels
- **ResNet**: Skip connections

Let's implement CNN operations!
    `,problems:["conv-output-size","conv2d-forward","max-pool","flatten-layer","conv2d-advanced"]},{id:"transformers",title:"Attention & Transformers",description:"Master attention mechanisms and transformer architecture.",icon:"🤖",introduction:`
# Attention and Transformers

Transformers have revolutionized NLP and are now used in vision, speech, and more.

## Key Concepts

### Self-Attention
- **Query, Key, Value**: Three projections of input
- **Attention scores**: Q @ K.T / sqrt(d_k)
- **Output**: Weighted sum of values

### Scaled Dot-Product Attention
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

### Multi-Head Attention
- **Multiple heads**: Parallel attention with different projections
- **Concatenate**: Combine all head outputs
- **Project**: Linear transformation back to model dimension

### Transformer Components
- **Positional Encoding**: Inject position information
- **Layer Normalization**: Normalize across features
- **Feed-Forward Network**: MLP after attention
- **Residual Connections**: Add input to output

### Causal Masking
- **Purpose**: Prevent attending to future tokens
- **Implementation**: Add -inf to upper triangle before softmax

### Interview Essentials
- Implement scaled dot-product attention
- Explain why we scale by sqrt(d_k)
- Multi-head attention benefits
- Positional encoding purpose

Let's build transformers from scratch!
    `,problems:["scaled-dot-product-attention","multi-head-attention","positional-encoding","layer-norm","causal-mask"]},{id:"generative-models",title:"Generative Models",description:"Learn VAEs, diffusion models, and generative AI fundamentals.",icon:"🎨",introduction:`
# Generative Models

Generative models learn to create new data similar to the training distribution.

## Key Concepts

### Variational Autoencoders (VAEs)
- **Encoder**: Maps input to latent distribution (μ, σ)
- **Decoder**: Reconstructs input from latent sample
- **Reparameterization trick**: Enable backprop through sampling
- **Loss**: Reconstruction + KL divergence

### VAE Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_reconstructed||² + KL(q(z|x) || p(z))
\`\`\`

### Diffusion Models
- **Forward process**: Gradually add noise to data
- **Reverse process**: Learn to denoise
- **Noise schedule**: β_t controls noise at each step

### Key Formulas
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### KL Divergence
- Measures difference between distributions
- KL(P || Q) ≥ 0, equals 0 iff P = Q
- Not symmetric

### Interview Topics
- Explain reparameterization trick
- VAE loss components
- Diffusion forward/reverse process
- Compare GANs vs VAEs vs Diffusion

Let's implement generative models!
    `,problems:["kl-divergence","vae-reparameterization","vae-loss","vqvae-quantization","diffusion-noise-schedule","diffusion-forward"]},{id:"e2e-implementations",title:"End-to-End Implementations",description:"Build complete ML models from scratch.",icon:"🏗️",introduction:`
# End-to-End Implementations

Put everything together to build complete models from scratch.

## Why Build From Scratch?

1. **Deep Understanding**: Know every component
2. **Interview Preparation**: Common coding questions
3. **Debugging Skills**: Understand what can go wrong
4. **Framework Independence**: Adapt to any library

## Models Covered

### 2-Layer MLP with Backprop
Complete neural network with:
- Forward pass through hidden layer
- Backpropagation of gradients
- Weight updates via gradient descent

### Transformer Encoder
Full encoder block with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Feed-forward network
- Residual connections

### Variational Autoencoder (VAE)
Generative model with:
- Encoder → latent distribution
- Reparameterization trick
- Decoder → reconstruction
- ELBO loss (reconstruction + KL)

### Diffusion Model
DDPM-style diffusion with:
- Noise schedule (β, α, ᾱ)
- Forward diffusion process
- Reverse denoising process
- Training objective

### Convolutional Neural Network
Image classifier with:
- Conv2D layers
- Max pooling
- Flatten + FC layers
- Full forward pass

Build these models to truly understand deep learning!
    `,problems:["e2e-mlp","e2e-transformer","e2e-vae","e2e-vqvae","e2e-diffusion","e2e-cnn"]}];function c0(){const{getOverallProgress:e}=Mi(),t=Sr.map(r=>({id:r.id,problemCount:r.problems.length})),n=e(t);return x.jsx("header",{className:"bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm",children:x.jsx("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",children:x.jsxs("div",{className:"flex items-center justify-between h-16",children:[x.jsxs(yn,{to:"/",className:"flex items-center gap-3",children:[x.jsx("div",{className:"w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center",children:x.jsx("span",{className:"text-white font-bold text-lg",children:"ML"})}),x.jsx("span",{className:"text-xl font-semibold text-gray-900",children:"ML Coding Lab"})]}),x.jsx("div",{className:"flex items-center gap-6",children:x.jsxs("div",{className:"flex items-center gap-3",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Overall Progress"}),x.jsx("div",{className:"w-32 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${n}%`}})}),x.jsxs("span",{className:"text-primary-600 text-sm font-medium",children:[n,"%"]})]})})]})})})}function d0(){const{getSectionProgress:e}=Mi();return x.jsx("aside",{className:"w-64 bg-white border-r border-gray-200 min-h-[calc(100vh-4rem)] p-4",children:x.jsxs("nav",{className:"space-y-2",children:[x.jsx("h3",{className:"text-gray-500 text-xs font-semibold uppercase tracking-wider mb-4 px-3",children:"Learning Sections"}),Sr.map((t,n)=>{const r=e(t.id,t.problems.length);return x.jsxs(a0,{to:`/section/${t.id}`,className:({isActive:i})=>`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${i?"bg-primary-50 text-primary-700":"text-gray-700 hover:bg-gray-100 hover:text-gray-900"}`,children:[x.jsx("span",{className:"w-6 h-6 flex items-center justify-center text-lg",children:t.icon}),x.jsxs("div",{className:"flex-1 min-w-0",children:[x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsxs("span",{className:"text-gray-400 text-xs",children:[n+1,"."]}),x.jsx("span",{className:"text-sm font-medium truncate",children:t.title})]}),x.jsxs("div",{className:"flex items-center gap-2 mt-1",children:[x.jsx("div",{className:"flex-1 h-1 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${r}%`}})}),x.jsxs("span",{className:"text-xs text-gray-400",children:[r,"%"]})]})]})]},t.id)})]})})}function p0({children:e}){return x.jsxs("div",{className:"min-h-screen bg-gray-50",children:[x.jsx(c0,{}),x.jsxs("div",{className:"flex",children:[x.jsx(d0,{}),x.jsx("main",{className:"flex-1 p-6",children:e})]})]})}function f0(){const{getSectionProgress:e,getOverallProgress:t}=Mi(),n=Sr.map(i=>({id:i.id,problemCount:i.problems.length})),r=t(n);return x.jsxs("div",{className:"max-w-5xl mx-auto",children:[x.jsxs("div",{className:"text-center mb-12",children:[x.jsx("h1",{className:"text-4xl font-bold text-gray-900 mb-4",children:"Learn ML By Building"}),x.jsx("p",{className:"text-lg text-gray-600 max-w-2xl mx-auto mb-8",children:"Practice hands-on machine learning coding problems. Run Python code directly in your browser with NumPy and scikit-learn support."}),x.jsxs("div",{className:"inline-flex items-center gap-4 bg-white rounded-lg px-6 py-3 shadow-sm border border-gray-200",children:[x.jsx("span",{className:"text-gray-600",children:"Overall Progress"}),x.jsx("div",{className:"w-48 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-500",style:{width:`${r}%`}})}),x.jsxs("span",{className:"text-primary-600 font-medium",children:[r,"%"]})]})]}),x.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:Sr.map((i,a)=>{const o=e(i.id,i.problems.length);return x.jsx(yn,{to:`/section/${i.id}`,className:"group bg-white rounded-xl p-6 border border-gray-200 hover:border-primary-400 hover:shadow-md transition-all duration-200",children:x.jsxs("div",{className:"flex items-start gap-4",children:[x.jsx("div",{className:"w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center text-2xl group-hover:bg-primary-50 transition-colors",children:i.icon}),x.jsxs("div",{className:"flex-1",children:[x.jsxs("div",{className:"flex items-center gap-2 mb-1",children:[x.jsxs("span",{className:"text-gray-400 text-sm",children:[a+1,"."]}),x.jsx("h3",{className:"text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors",children:i.title})]}),x.jsx("p",{className:"text-gray-500 text-sm mb-4",children:i.description}),x.jsxs("div",{className:"flex items-center justify-between",children:[x.jsxs("span",{className:"text-gray-400 text-sm",children:[i.problems.length," problems"]}),x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("div",{className:"w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${o}%`}})}),x.jsxs("span",{className:"text-xs text-gray-500",children:[o,"%"]})]})]})]})]})},i.id)})}),x.jsxs("div",{className:"mt-16 grid grid-cols-1 md:grid-cols-3 gap-6",children:[x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"🐍"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Python in Browser"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Run Python code with NumPy, pandas, and more - no setup required."})]}),x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"✅"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Instant Feedback"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Test your code against multiple test cases and see results immediately."})]}),x.jsxs("div",{className:"bg-white rounded-lg p-6 text-center border border-gray-200",children:[x.jsx("div",{className:"text-3xl mb-3",children:"📊"}),x.jsx("h3",{className:"text-gray-900 font-medium mb-2",children:"Track Progress"}),x.jsx("p",{className:"text-gray-500 text-sm",children:"Your progress is saved locally. Pick up where you left off anytime."})]})]})]})}function m0(e,t){const n={};return(e[e.length-1]===""?[...e,""]:e).join((n.padRight?" ":"")+","+(n.padLeft===!1?"":" ")).trim()}const h0=/^[$_\p{ID_Start}][$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,g0=/^[$_\p{ID_Start}][-$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,y0={};function rd(e,t){return(y0.jsx?g0:h0).test(e)}const x0=/[ \t\n\f\r]/g;function _0(e){return typeof e=="object"?e.type==="text"?id(e.value):!1:id(e)}function id(e){return e.replace(x0,"")===""}class Di{constructor(t,n,r){this.normal=n,this.property=t,r&&(this.space=r)}}Di.prototype.normal={};Di.prototype.property={};Di.prototype.space=void 0;function Pm(e,t){const n={},r={};for(const i of e)Object.assign(n,i.property),Object.assign(r,i.normal);return new Di(n,r,t)}function gl(e){return e.toLowerCase()}class Ge{constructor(t,n){this.attribute=n,this.property=t}}Ge.prototype.attribute="";Ge.prototype.booleanish=!1;Ge.prototype.boolean=!1;Ge.prototype.commaOrSpaceSeparated=!1;Ge.prototype.commaSeparated=!1;Ge.prototype.defined=!1;Ge.prototype.mustUseProperty=!1;Ge.prototype.number=!1;Ge.prototype.overloadedBoolean=!1;Ge.prototype.property="";Ge.prototype.spaceSeparated=!1;Ge.prototype.space=void 0;let v0=0;const G=Hn(),be=Hn(),yl=Hn(),R=Hn(),ue=Hn(),mr=Hn(),Je=Hn();function Hn(){return 2**++v0}const xl=Object.freeze(Object.defineProperty({__proto__:null,boolean:G,booleanish:be,commaOrSpaceSeparated:Je,commaSeparated:mr,number:R,overloadedBoolean:yl,spaceSeparated:ue},Symbol.toStringTag,{value:"Module"})),rs=Object.keys(xl);class ku extends Ge{constructor(t,n,r,i){let a=-1;if(super(t,n),ad(this,"space",i),typeof r=="number")for(;++a<rs.length;){const o=rs[a];ad(this,rs[a],(r&xl[o])===xl[o])}}}ku.prototype.defined=!0;function ad(e,t,n){n&&(e[t]=n)}function Nr(e){const t={},n={};for(const[r,i]of Object.entries(e.properties)){const a=new ku(r,e.transform(e.attributes||{},r),i,e.space);e.mustUseProperty&&e.mustUseProperty.includes(r)&&(a.mustUseProperty=!0),t[r]=a,n[gl(r)]=r,n[gl(a.attribute)]=r}return new Di(t,n,e.space)}const Nm=Nr({properties:{ariaActiveDescendant:null,ariaAtomic:be,ariaAutoComplete:null,ariaBusy:be,ariaChecked:be,ariaColCount:R,ariaColIndex:R,ariaColSpan:R,ariaControls:ue,ariaCurrent:null,ariaDescribedBy:ue,ariaDetails:null,ariaDisabled:be,ariaDropEffect:ue,ariaErrorMessage:null,ariaExpanded:be,ariaFlowTo:ue,ariaGrabbed:be,ariaHasPopup:null,ariaHidden:be,ariaInvalid:null,ariaKeyShortcuts:null,ariaLabel:null,ariaLabelledBy:ue,ariaLevel:R,ariaLive:null,ariaModal:be,ariaMultiLine:be,ariaMultiSelectable:be,ariaOrientation:null,ariaOwns:ue,ariaPlaceholder:null,ariaPosInSet:R,ariaPressed:be,ariaReadOnly:be,ariaRelevant:null,ariaRequired:be,ariaRoleDescription:ue,ariaRowCount:R,ariaRowIndex:R,ariaRowSpan:R,ariaSelected:be,ariaSetSize:R,ariaSort:null,ariaValueMax:R,ariaValueMin:R,ariaValueNow:R,ariaValueText:null,role:null},transform(e,t){return t==="role"?t:"aria-"+t.slice(4).toLowerCase()}});function Tm(e,t){return t in e?e[t]:t}function Lm(e,t){return Tm(e,t.toLowerCase())}const w0=Nr({attributes:{acceptcharset:"accept-charset",classname:"class",htmlfor:"for",httpequiv:"http-equiv"},mustUseProperty:["checked","multiple","muted","selected"],properties:{abbr:null,accept:mr,acceptCharset:ue,accessKey:ue,action:null,allow:null,allowFullScreen:G,allowPaymentRequest:G,allowUserMedia:G,alt:null,as:null,async:G,autoCapitalize:null,autoComplete:ue,autoFocus:G,autoPlay:G,blocking:ue,capture:null,charSet:null,checked:G,cite:null,className:ue,cols:R,colSpan:null,content:null,contentEditable:be,controls:G,controlsList:ue,coords:R|mr,crossOrigin:null,data:null,dateTime:null,decoding:null,default:G,defer:G,dir:null,dirName:null,disabled:G,download:yl,draggable:be,encType:null,enterKeyHint:null,fetchPriority:null,form:null,formAction:null,formEncType:null,formMethod:null,formNoValidate:G,formTarget:null,headers:ue,height:R,hidden:yl,high:R,href:null,hrefLang:null,htmlFor:ue,httpEquiv:ue,id:null,imageSizes:null,imageSrcSet:null,inert:G,inputMode:null,integrity:null,is:null,isMap:G,itemId:null,itemProp:ue,itemRef:ue,itemScope:G,itemType:ue,kind:null,label:null,lang:null,language:null,list:null,loading:null,loop:G,low:R,manifest:null,max:null,maxLength:R,media:null,method:null,min:null,minLength:R,multiple:G,muted:G,name:null,nonce:null,noModule:G,noValidate:G,onAbort:null,onAfterPrint:null,onAuxClick:null,onBeforeMatch:null,onBeforePrint:null,onBeforeToggle:null,onBeforeUnload:null,onBlur:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onContextLost:null,onContextMenu:null,onContextRestored:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnded:null,onError:null,onFocus:null,onFormData:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLanguageChange:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadEnd:null,onLoadStart:null,onMessage:null,onMessageError:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRejectionHandled:null,onReset:null,onResize:null,onScroll:null,onScrollEnd:null,onSecurityPolicyViolation:null,onSeeked:null,onSeeking:null,onSelect:null,onSlotChange:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnhandledRejection:null,onUnload:null,onVolumeChange:null,onWaiting:null,onWheel:null,open:G,optimum:R,pattern:null,ping:ue,placeholder:null,playsInline:G,popover:null,popoverTarget:null,popoverTargetAction:null,poster:null,preload:null,readOnly:G,referrerPolicy:null,rel:ue,required:G,reversed:G,rows:R,rowSpan:R,sandbox:ue,scope:null,scoped:G,seamless:G,selected:G,shadowRootClonable:G,shadowRootDelegatesFocus:G,shadowRootMode:null,shape:null,size:R,sizes:null,slot:null,span:R,spellCheck:be,src:null,srcDoc:null,srcLang:null,srcSet:null,start:R,step:null,style:null,tabIndex:R,target:null,title:null,translate:null,type:null,typeMustMatch:G,useMap:null,value:be,width:R,wrap:null,writingSuggestions:null,align:null,aLink:null,archive:ue,axis:null,background:null,bgColor:null,border:R,borderColor:null,bottomMargin:R,cellPadding:null,cellSpacing:null,char:null,charOff:null,classId:null,clear:null,code:null,codeBase:null,codeType:null,color:null,compact:G,declare:G,event:null,face:null,frame:null,frameBorder:null,hSpace:R,leftMargin:R,link:null,longDesc:null,lowSrc:null,marginHeight:R,marginWidth:R,noResize:G,noHref:G,noShade:G,noWrap:G,object:null,profile:null,prompt:null,rev:null,rightMargin:R,rules:null,scheme:null,scrolling:be,standby:null,summary:null,text:null,topMargin:R,valueType:null,version:null,vAlign:null,vLink:null,vSpace:R,allowTransparency:null,autoCorrect:null,autoSave:null,disablePictureInPicture:G,disableRemotePlayback:G,prefix:null,property:null,results:R,security:null,unselectable:null},space:"html",transform:Lm}),k0=Nr({attributes:{accentHeight:"accent-height",alignmentBaseline:"alignment-baseline",arabicForm:"arabic-form",baselineShift:"baseline-shift",capHeight:"cap-height",className:"class",clipPath:"clip-path",clipRule:"clip-rule",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",crossOrigin:"crossorigin",dataType:"datatype",dominantBaseline:"dominant-baseline",enableBackground:"enable-background",fillOpacity:"fill-opacity",fillRule:"fill-rule",floodColor:"flood-color",floodOpacity:"flood-opacity",fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",hrefLang:"hreflang",horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",horizOriginY:"horiz-origin-y",imageRendering:"image-rendering",letterSpacing:"letter-spacing",lightingColor:"lighting-color",markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",navDown:"nav-down",navDownLeft:"nav-down-left",navDownRight:"nav-down-right",navLeft:"nav-left",navNext:"nav-next",navPrev:"nav-prev",navRight:"nav-right",navUp:"nav-up",navUpLeft:"nav-up-left",navUpRight:"nav-up-right",onAbort:"onabort",onActivate:"onactivate",onAfterPrint:"onafterprint",onBeforePrint:"onbeforeprint",onBegin:"onbegin",onCancel:"oncancel",onCanPlay:"oncanplay",onCanPlayThrough:"oncanplaythrough",onChange:"onchange",onClick:"onclick",onClose:"onclose",onCopy:"oncopy",onCueChange:"oncuechange",onCut:"oncut",onDblClick:"ondblclick",onDrag:"ondrag",onDragEnd:"ondragend",onDragEnter:"ondragenter",onDragExit:"ondragexit",onDragLeave:"ondragleave",onDragOver:"ondragover",onDragStart:"ondragstart",onDrop:"ondrop",onDurationChange:"ondurationchange",onEmptied:"onemptied",onEnd:"onend",onEnded:"onended",onError:"onerror",onFocus:"onfocus",onFocusIn:"onfocusin",onFocusOut:"onfocusout",onHashChange:"onhashchange",onInput:"oninput",onInvalid:"oninvalid",onKeyDown:"onkeydown",onKeyPress:"onkeypress",onKeyUp:"onkeyup",onLoad:"onload",onLoadedData:"onloadeddata",onLoadedMetadata:"onloadedmetadata",onLoadStart:"onloadstart",onMessage:"onmessage",onMouseDown:"onmousedown",onMouseEnter:"onmouseenter",onMouseLeave:"onmouseleave",onMouseMove:"onmousemove",onMouseOut:"onmouseout",onMouseOver:"onmouseover",onMouseUp:"onmouseup",onMouseWheel:"onmousewheel",onOffline:"onoffline",onOnline:"ononline",onPageHide:"onpagehide",onPageShow:"onpageshow",onPaste:"onpaste",onPause:"onpause",onPlay:"onplay",onPlaying:"onplaying",onPopState:"onpopstate",onProgress:"onprogress",onRateChange:"onratechange",onRepeat:"onrepeat",onReset:"onreset",onResize:"onresize",onScroll:"onscroll",onSeeked:"onseeked",onSeeking:"onseeking",onSelect:"onselect",onShow:"onshow",onStalled:"onstalled",onStorage:"onstorage",onSubmit:"onsubmit",onSuspend:"onsuspend",onTimeUpdate:"ontimeupdate",onToggle:"ontoggle",onUnload:"onunload",onVolumeChange:"onvolumechange",onWaiting:"onwaiting",onZoom:"onzoom",overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pointerEvents:"pointer-events",referrerPolicy:"referrerpolicy",renderingIntent:"rendering-intent",shapeRendering:"shape-rendering",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",strokeDashArray:"stroke-dasharray",strokeDashOffset:"stroke-dashoffset",strokeLineCap:"stroke-linecap",strokeLineJoin:"stroke-linejoin",strokeMiterLimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",tabIndex:"tabindex",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",transformOrigin:"transform-origin",typeOf:"typeof",underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",vectorEffect:"vector-effect",vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",wordSpacing:"word-spacing",writingMode:"writing-mode",xHeight:"x-height",playbackOrder:"playbackorder",timelineBegin:"timelinebegin"},properties:{about:Je,accentHeight:R,accumulate:null,additive:null,alignmentBaseline:null,alphabetic:R,amplitude:R,arabicForm:null,ascent:R,attributeName:null,attributeType:null,azimuth:R,bandwidth:null,baselineShift:null,baseFrequency:null,baseProfile:null,bbox:null,begin:null,bias:R,by:null,calcMode:null,capHeight:R,className:ue,clip:null,clipPath:null,clipPathUnits:null,clipRule:null,color:null,colorInterpolation:null,colorInterpolationFilters:null,colorProfile:null,colorRendering:null,content:null,contentScriptType:null,contentStyleType:null,crossOrigin:null,cursor:null,cx:null,cy:null,d:null,dataType:null,defaultAction:null,descent:R,diffuseConstant:R,direction:null,display:null,dur:null,divisor:R,dominantBaseline:null,download:G,dx:null,dy:null,edgeMode:null,editable:null,elevation:R,enableBackground:null,end:null,event:null,exponent:R,externalResourcesRequired:null,fill:null,fillOpacity:R,fillRule:null,filter:null,filterRes:null,filterUnits:null,floodColor:null,floodOpacity:null,focusable:null,focusHighlight:null,fontFamily:null,fontSize:null,fontSizeAdjust:null,fontStretch:null,fontStyle:null,fontVariant:null,fontWeight:null,format:null,fr:null,from:null,fx:null,fy:null,g1:mr,g2:mr,glyphName:mr,glyphOrientationHorizontal:null,glyphOrientationVertical:null,glyphRef:null,gradientTransform:null,gradientUnits:null,handler:null,hanging:R,hatchContentUnits:null,hatchUnits:null,height:null,href:null,hrefLang:null,horizAdvX:R,horizOriginX:R,horizOriginY:R,id:null,ideographic:R,imageRendering:null,initialVisibility:null,in:null,in2:null,intercept:R,k:R,k1:R,k2:R,k3:R,k4:R,kernelMatrix:Je,kernelUnitLength:null,keyPoints:null,keySplines:null,keyTimes:null,kerning:null,lang:null,lengthAdjust:null,letterSpacing:null,lightingColor:null,limitingConeAngle:R,local:null,markerEnd:null,markerMid:null,markerStart:null,markerHeight:null,markerUnits:null,markerWidth:null,mask:null,maskContentUnits:null,maskUnits:null,mathematical:null,max:null,media:null,mediaCharacterEncoding:null,mediaContentEncodings:null,mediaSize:R,mediaTime:null,method:null,min:null,mode:null,name:null,navDown:null,navDownLeft:null,navDownRight:null,navLeft:null,navNext:null,navPrev:null,navRight:null,navUp:null,navUpLeft:null,navUpRight:null,numOctaves:null,observer:null,offset:null,onAbort:null,onActivate:null,onAfterPrint:null,onBeforePrint:null,onBegin:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnd:null,onEnded:null,onError:null,onFocus:null,onFocusIn:null,onFocusOut:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadStart:null,onMessage:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onMouseWheel:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRepeat:null,onReset:null,onResize:null,onScroll:null,onSeeked:null,onSeeking:null,onSelect:null,onShow:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnload:null,onVolumeChange:null,onWaiting:null,onZoom:null,opacity:null,operator:null,order:null,orient:null,orientation:null,origin:null,overflow:null,overlay:null,overlinePosition:R,overlineThickness:R,paintOrder:null,panose1:null,path:null,pathLength:R,patternContentUnits:null,patternTransform:null,patternUnits:null,phase:null,ping:ue,pitch:null,playbackOrder:null,pointerEvents:null,points:null,pointsAtX:R,pointsAtY:R,pointsAtZ:R,preserveAlpha:null,preserveAspectRatio:null,primitiveUnits:null,propagate:null,property:Je,r:null,radius:null,referrerPolicy:null,refX:null,refY:null,rel:Je,rev:Je,renderingIntent:null,repeatCount:null,repeatDur:null,requiredExtensions:Je,requiredFeatures:Je,requiredFonts:Je,requiredFormats:Je,resource:null,restart:null,result:null,rotate:null,rx:null,ry:null,scale:null,seed:null,shapeRendering:null,side:null,slope:null,snapshotTime:null,specularConstant:R,specularExponent:R,spreadMethod:null,spacing:null,startOffset:null,stdDeviation:null,stemh:null,stemv:null,stitchTiles:null,stopColor:null,stopOpacity:null,strikethroughPosition:R,strikethroughThickness:R,string:null,stroke:null,strokeDashArray:Je,strokeDashOffset:null,strokeLineCap:null,strokeLineJoin:null,strokeMiterLimit:R,strokeOpacity:R,strokeWidth:null,style:null,surfaceScale:R,syncBehavior:null,syncBehaviorDefault:null,syncMaster:null,syncTolerance:null,syncToleranceDefault:null,systemLanguage:Je,tabIndex:R,tableValues:null,target:null,targetX:R,targetY:R,textAnchor:null,textDecoration:null,textRendering:null,textLength:null,timelineBegin:null,title:null,transformBehavior:null,type:null,typeOf:Je,to:null,transform:null,transformOrigin:null,u1:null,u2:null,underlinePosition:R,underlineThickness:R,unicode:null,unicodeBidi:null,unicodeRange:null,unitsPerEm:R,values:null,vAlphabetic:R,vMathematical:R,vectorEffect:null,vHanging:R,vIdeographic:R,version:null,vertAdvY:R,vertOriginX:R,vertOriginY:R,viewBox:null,viewTarget:null,visibility:null,width:null,widths:null,wordSpacing:null,writingMode:null,x:null,x1:null,x2:null,xChannelSelector:null,xHeight:R,y:null,y1:null,y2:null,yChannelSelector:null,z:null,zoomAndPan:null},space:"svg",transform:Tm}),Am=Nr({properties:{xLinkActuate:null,xLinkArcRole:null,xLinkHref:null,xLinkRole:null,xLinkShow:null,xLinkTitle:null,xLinkType:null},space:"xlink",transform(e,t){return"xlink:"+t.slice(5).toLowerCase()}}),Rm=Nr({attributes:{xmlnsxlink:"xmlns:xlink"},properties:{xmlnsXLink:null,xmlns:null},space:"xmlns",transform:Lm}),Im=Nr({properties:{xmlBase:null,xmlLang:null,xmlSpace:null},space:"xml",transform(e,t){return"xml:"+t.slice(3).toLowerCase()}}),b0={classId:"classID",dataType:"datatype",itemId:"itemID",strokeDashArray:"strokeDasharray",strokeDashOffset:"strokeDashoffset",strokeLineCap:"strokeLinecap",strokeLineJoin:"strokeLinejoin",strokeMiterLimit:"strokeMiterlimit",typeOf:"typeof",xLinkActuate:"xlinkActuate",xLinkArcRole:"xlinkArcrole",xLinkHref:"xlinkHref",xLinkRole:"xlinkRole",xLinkShow:"xlinkShow",xLinkTitle:"xlinkTitle",xLinkType:"xlinkType",xmlnsXLink:"xmlnsXlink"},S0=/[A-Z]/g,od=/-[a-z]/g,C0=/^data[-\w.:]+$/i;function z0(e,t){const n=gl(t);let r=t,i=Ge;if(n in e.normal)return e.property[e.normal[n]];if(n.length>4&&n.slice(0,4)==="data"&&C0.test(t)){if(t.charAt(4)==="-"){const a=t.slice(5).replace(od,P0);r="data"+a.charAt(0).toUpperCase()+a.slice(1)}else{const a=t.slice(4);if(!od.test(a)){let o=a.replace(S0,E0);o.charAt(0)!=="-"&&(o="-"+o),t="data"+o}}i=ku}return new i(r,t)}function E0(e){return"-"+e.toLowerCase()}function P0(e){return e.charAt(1).toUpperCase()}const N0=Pm([Nm,w0,Am,Rm,Im],"html"),bu=Pm([Nm,k0,Am,Rm,Im],"svg");function T0(e){return e.join(" ").trim()}var Su={},sd=/\/\*[^*]*\*+([^/*][^*]*\*+)*\//g,L0=/\n/g,A0=/^\s*/,R0=/^(\*?[-#/*\\\w]+(\[[0-9a-z_-]+\])?)\s*/,I0=/^:\s*/,j0=/^((?:'(?:\\'|.)*?'|"(?:\\"|.)*?"|\([^)]*?\)|[^};])+)/,O0=/^[;\s]*/,M0=/^\s+|\s+$/g,D0=`
`,ld="/",ud="*",An="",F0="comment",W0="declaration";function B0(e,t){if(typeof e!="string")throw new TypeError("First argument must be a string");if(!e)return[];t=t||{};var n=1,r=1;function i(w){var _=w.match(L0);_&&(n+=_.length);var S=w.lastIndexOf(D0);r=~S?w.length-S:r+w.length}function a(){var w={line:n,column:r};return function(_){return _.position=new o(w),u(),_}}function o(w){this.start=w,this.end={line:n,column:r},this.source=t.source}o.prototype.content=e;function s(w){var _=new Error(t.source+":"+n+":"+r+": "+w);if(_.reason=w,_.filename=t.source,_.line=n,_.column=r,_.source=e,!t.silent)throw _}function l(w){var _=w.exec(e);if(_){var S=_[0];return i(S),e=e.slice(S.length),_}}function u(){l(A0)}function c(w){var _;for(w=w||[];_=d();)_!==!1&&w.push(_);return w}function d(){var w=a();if(!(ld!=e.charAt(0)||ud!=e.charAt(1))){for(var _=2;An!=e.charAt(_)&&(ud!=e.charAt(_)||ld!=e.charAt(_+1));)++_;if(_+=2,An===e.charAt(_-1))return s("End of comment missing");var S=e.slice(2,_-2);return r+=2,i(S),e=e.slice(_),r+=2,w({type:F0,comment:S})}}function p(){var w=a(),_=l(R0);if(_){if(d(),!l(I0))return s("property missing ':'");var S=l(j0),h=w({type:W0,property:cd(_[0].replace(sd,An)),value:S?cd(S[0].replace(sd,An)):An});return l(O0),h}}function f(){var w=[];c(w);for(var _;_=p();)_!==!1&&(w.push(_),c(w));return w}return u(),f()}function cd(e){return e?e.replace(M0,An):An}var q0=B0,U0=za&&za.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(Su,"__esModule",{value:!0});Su.default=H0;const V0=U0(q0);function H0(e,t){let n=null;if(!e||typeof e!="string")return n;const r=(0,V0.default)(e),i=typeof t=="function";return r.forEach(a=>{if(a.type!=="declaration")return;const{property:o,value:s}=a;i?t(o,s,a):s&&(n=n||{},n[o]=s)}),n}var So={};Object.defineProperty(So,"__esModule",{value:!0});So.camelCase=void 0;var $0=/^--[a-zA-Z0-9_-]+$/,Q0=/-([a-z])/g,X0=/^[^-]+$/,K0=/^-(webkit|moz|ms|o|khtml)-/,Y0=/^-(ms)-/,G0=function(e){return!e||X0.test(e)||$0.test(e)},Z0=function(e,t){return t.toUpperCase()},dd=function(e,t){return"".concat(t,"-")},J0=function(e,t){return t===void 0&&(t={}),G0(e)?e:(e=e.toLowerCase(),t.reactCompat?e=e.replace(Y0,dd):e=e.replace(K0,dd),e.replace(Q0,Z0))};So.camelCase=J0;var e_=za&&za.__importDefault||function(e){return e&&e.__esModule?e:{default:e}},t_=e_(Su),n_=So;function _l(e,t){var n={};return!e||typeof e!="string"||(0,t_.default)(e,function(r,i){r&&i&&(n[(0,n_.camelCase)(r,t)]=i)}),n}_l.default=_l;var r_=_l;const i_=ro(r_),jm=Om("end"),Cu=Om("start");function Om(e){return t;function t(n){const r=n&&n.position&&n.position[e]||{};if(typeof r.line=="number"&&r.line>0&&typeof r.column=="number"&&r.column>0)return{line:r.line,column:r.column,offset:typeof r.offset=="number"&&r.offset>-1?r.offset:void 0}}}function a_(e){const t=Cu(e),n=jm(e);if(t&&n)return{start:t,end:n}}function ii(e){return!e||typeof e!="object"?"":"position"in e||"type"in e?pd(e.position):"start"in e||"end"in e?pd(e):"line"in e||"column"in e?vl(e):""}function vl(e){return fd(e&&e.line)+":"+fd(e&&e.column)}function pd(e){return vl(e&&e.start)+"-"+vl(e&&e.end)}function fd(e){return e&&typeof e=="number"?e:1}class De extends Error{constructor(t,n,r){super(),typeof n=="string"&&(r=n,n=void 0);let i="",a={},o=!1;if(n&&("line"in n&&"column"in n?a={place:n}:"start"in n&&"end"in n?a={place:n}:"type"in n?a={ancestors:[n],place:n.position}:a={...n}),typeof t=="string"?i=t:!a.cause&&t&&(o=!0,i=t.message,a.cause=t),!a.ruleId&&!a.source&&typeof r=="string"){const l=r.indexOf(":");l===-1?a.ruleId=r:(a.source=r.slice(0,l),a.ruleId=r.slice(l+1))}if(!a.place&&a.ancestors&&a.ancestors){const l=a.ancestors[a.ancestors.length-1];l&&(a.place=l.position)}const s=a.place&&"start"in a.place?a.place.start:a.place;this.ancestors=a.ancestors||void 0,this.cause=a.cause||void 0,this.column=s?s.column:void 0,this.fatal=void 0,this.file="",this.message=i,this.line=s?s.line:void 0,this.name=ii(a.place)||"1:1",this.place=a.place||void 0,this.reason=this.message,this.ruleId=a.ruleId||void 0,this.source=a.source||void 0,this.stack=o&&a.cause&&typeof a.cause.stack=="string"?a.cause.stack:"",this.actual=void 0,this.expected=void 0,this.note=void 0,this.url=void 0}}De.prototype.file="";De.prototype.name="";De.prototype.reason="";De.prototype.message="";De.prototype.stack="";De.prototype.column=void 0;De.prototype.line=void 0;De.prototype.ancestors=void 0;De.prototype.cause=void 0;De.prototype.fatal=void 0;De.prototype.place=void 0;De.prototype.ruleId=void 0;De.prototype.source=void 0;const zu={}.hasOwnProperty,o_=new Map,s_=/[A-Z]/g,l_=new Set(["table","tbody","thead","tfoot","tr"]),u_=new Set(["td","th"]),Mm="https://github.com/syntax-tree/hast-util-to-jsx-runtime";function c_(e,t){if(!t||t.Fragment===void 0)throw new TypeError("Expected `Fragment` in options");const n=t.filePath||void 0;let r;if(t.development){if(typeof t.jsxDEV!="function")throw new TypeError("Expected `jsxDEV` in options when `development: true`");r=x_(n,t.jsxDEV)}else{if(typeof t.jsx!="function")throw new TypeError("Expected `jsx` in production options");if(typeof t.jsxs!="function")throw new TypeError("Expected `jsxs` in production options");r=y_(n,t.jsx,t.jsxs)}const i={Fragment:t.Fragment,ancestors:[],components:t.components||{},create:r,elementAttributeNameCase:t.elementAttributeNameCase||"react",evaluater:t.createEvaluater?t.createEvaluater():void 0,filePath:n,ignoreInvalidStyle:t.ignoreInvalidStyle||!1,passKeys:t.passKeys!==!1,passNode:t.passNode||!1,schema:t.space==="svg"?bu:N0,stylePropertyNameCase:t.stylePropertyNameCase||"dom",tableCellAlignToStyle:t.tableCellAlignToStyle!==!1},a=Dm(i,e,void 0);return a&&typeof a!="string"?a:i.create(e,i.Fragment,{children:a||void 0},void 0)}function Dm(e,t,n){if(t.type==="element")return d_(e,t,n);if(t.type==="mdxFlowExpression"||t.type==="mdxTextExpression")return p_(e,t);if(t.type==="mdxJsxFlowElement"||t.type==="mdxJsxTextElement")return m_(e,t,n);if(t.type==="mdxjsEsm")return f_(e,t);if(t.type==="root")return h_(e,t,n);if(t.type==="text")return g_(e,t)}function d_(e,t,n){const r=e.schema;let i=r;t.tagName.toLowerCase()==="svg"&&r.space==="html"&&(i=bu,e.schema=i),e.ancestors.push(t);const a=Wm(e,t.tagName,!1),o=__(e,t);let s=Pu(e,t);return l_.has(t.tagName)&&(s=s.filter(function(l){return typeof l=="string"?!_0(l):!0})),Fm(e,o,a,t),Eu(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function p_(e,t){if(t.data&&t.data.estree&&e.evaluater){const r=t.data.estree.body[0];return r.type,e.evaluater.evaluateExpression(r.expression)}Ei(e,t.position)}function f_(e,t){if(t.data&&t.data.estree&&e.evaluater)return e.evaluater.evaluateProgram(t.data.estree);Ei(e,t.position)}function m_(e,t,n){const r=e.schema;let i=r;t.name==="svg"&&r.space==="html"&&(i=bu,e.schema=i),e.ancestors.push(t);const a=t.name===null?e.Fragment:Wm(e,t.name,!0),o=v_(e,t),s=Pu(e,t);return Fm(e,o,a,t),Eu(o,s),e.ancestors.pop(),e.schema=r,e.create(t,a,o,n)}function h_(e,t,n){const r={};return Eu(r,Pu(e,t)),e.create(t,e.Fragment,r,n)}function g_(e,t){return t.value}function Fm(e,t,n,r){typeof n!="string"&&n!==e.Fragment&&e.passNode&&(t.node=r)}function Eu(e,t){if(t.length>0){const n=t.length>1?t:t[0];n&&(e.children=n)}}function y_(e,t,n){return r;function r(i,a,o,s){const u=Array.isArray(o.children)?n:t;return s?u(a,o,s):u(a,o)}}function x_(e,t){return n;function n(r,i,a,o){const s=Array.isArray(a.children),l=Cu(r);return t(i,a,o,s,{columnNumber:l?l.column-1:void 0,fileName:e,lineNumber:l?l.line:void 0},void 0)}}function __(e,t){const n={};let r,i;for(i in t.properties)if(i!=="children"&&zu.call(t.properties,i)){const a=w_(e,i,t.properties[i]);if(a){const[o,s]=a;e.tableCellAlignToStyle&&o==="align"&&typeof s=="string"&&u_.has(t.tagName)?r=s:n[o]=s}}if(r){const a=n.style||(n.style={});a[e.stylePropertyNameCase==="css"?"text-align":"textAlign"]=r}return n}function v_(e,t){const n={};for(const r of t.attributes)if(r.type==="mdxJsxExpressionAttribute")if(r.data&&r.data.estree&&e.evaluater){const a=r.data.estree.body[0];a.type;const o=a.expression;o.type;const s=o.properties[0];s.type,Object.assign(n,e.evaluater.evaluateExpression(s.argument))}else Ei(e,t.position);else{const i=r.name;let a;if(r.value&&typeof r.value=="object")if(r.value.data&&r.value.data.estree&&e.evaluater){const s=r.value.data.estree.body[0];s.type,a=e.evaluater.evaluateExpression(s.expression)}else Ei(e,t.position);else a=r.value===null?!0:r.value;n[i]=a}return n}function Pu(e,t){const n=[];let r=-1;const i=e.passKeys?new Map:o_;for(;++r<t.children.length;){const a=t.children[r];let o;if(e.passKeys){const l=a.type==="element"?a.tagName:a.type==="mdxJsxFlowElement"||a.type==="mdxJsxTextElement"?a.name:void 0;if(l){const u=i.get(l)||0;o=l+"-"+u,i.set(l,u+1)}}const s=Dm(e,a,o);s!==void 0&&n.push(s)}return n}function w_(e,t,n){const r=z0(e.schema,t);if(!(n==null||typeof n=="number"&&Number.isNaN(n))){if(Array.isArray(n)&&(n=r.commaSeparated?m0(n):T0(n)),r.property==="style"){let i=typeof n=="object"?n:k_(e,String(n));return e.stylePropertyNameCase==="css"&&(i=b_(i)),["style",i]}return[e.elementAttributeNameCase==="react"&&r.space?b0[r.property]||r.property:r.attribute,n]}}function k_(e,t){try{return i_(t,{reactCompat:!0})}catch(n){if(e.ignoreInvalidStyle)return{};const r=n,i=new De("Cannot parse `style` attribute",{ancestors:e.ancestors,cause:r,ruleId:"style",source:"hast-util-to-jsx-runtime"});throw i.file=e.filePath||void 0,i.url=Mm+"#cannot-parse-style-attribute",i}}function Wm(e,t,n){let r;if(!n)r={type:"Literal",value:t};else if(t.includes(".")){const i=t.split(".");let a=-1,o;for(;++a<i.length;){const s=rd(i[a])?{type:"Identifier",name:i[a]}:{type:"Literal",value:i[a]};o=o?{type:"MemberExpression",object:o,property:s,computed:!!(a&&s.type==="Literal"),optional:!1}:s}r=o}else r=rd(t)&&!/^[a-z]/.test(t)?{type:"Identifier",name:t}:{type:"Literal",value:t};if(r.type==="Literal"){const i=r.value;return zu.call(e.components,i)?e.components[i]:i}if(e.evaluater)return e.evaluater.evaluateExpression(r);Ei(e)}function Ei(e,t){const n=new De("Cannot handle MDX estrees without `createEvaluater`",{ancestors:e.ancestors,place:t,ruleId:"mdx-estree",source:"hast-util-to-jsx-runtime"});throw n.file=e.filePath||void 0,n.url=Mm+"#cannot-handle-mdx-estrees-without-createevaluater",n}function b_(e){const t={};let n;for(n in e)zu.call(e,n)&&(t[S_(n)]=e[n]);return t}function S_(e){let t=e.replace(s_,C_);return t.slice(0,3)==="ms-"&&(t="-"+t),t}function C_(e){return"-"+e.toLowerCase()}const is={action:["form"],cite:["blockquote","del","ins","q"],data:["object"],formAction:["button","input"],href:["a","area","base","link"],icon:["menuitem"],itemId:null,manifest:["html"],ping:["a","area"],poster:["video"],src:["audio","embed","iframe","img","input","script","source","track","video"]},z_={};function E_(e,t){const n=z_,r=typeof n.includeImageAlt=="boolean"?n.includeImageAlt:!0,i=typeof n.includeHtml=="boolean"?n.includeHtml:!0;return Bm(e,r,i)}function Bm(e,t,n){if(P_(e)){if("value"in e)return e.type==="html"&&!n?"":e.value;if(t&&"alt"in e&&e.alt)return e.alt;if("children"in e)return md(e.children,t,n)}return Array.isArray(e)?md(e,t,n):""}function md(e,t,n){const r=[];let i=-1;for(;++i<e.length;)r[i]=Bm(e[i],t,n);return r.join("")}function P_(e){return!!(e&&typeof e=="object")}const hd=document.createElement("i");function Nu(e){const t="&"+e+";";hd.innerHTML=t;const n=hd.textContent;return n.charCodeAt(n.length-1)===59&&e!=="semi"||n===t?!1:n}function Dt(e,t,n,r){const i=e.length;let a=0,o;if(t<0?t=-t>i?0:i+t:t=t>i?i:t,n=n>0?n:0,r.length<1e4)o=Array.from(r),o.unshift(t,n),e.splice(...o);else for(n&&e.splice(t,n);a<r.length;)o=r.slice(a,a+1e4),o.unshift(t,0),e.splice(...o),a+=1e4,t+=1e4}function pt(e,t){return e.length>0?(Dt(e,e.length,0,t),e):t}const gd={}.hasOwnProperty;function N_(e){const t={};let n=-1;for(;++n<e.length;)T_(t,e[n]);return t}function T_(e,t){let n;for(n in t){const i=(gd.call(e,n)?e[n]:void 0)||(e[n]={}),a=t[n];let o;if(a)for(o in a){gd.call(i,o)||(i[o]=[]);const s=a[o];L_(i[o],Array.isArray(s)?s:s?[s]:[])}}}function L_(e,t){let n=-1;const r=[];for(;++n<t.length;)(t[n].add==="after"?e:r).push(t[n]);Dt(e,0,0,r)}function qm(e,t){const n=Number.parseInt(e,t);return n<9||n===11||n>13&&n<32||n>126&&n<160||n>55295&&n<57344||n>64975&&n<65008||(n&65535)===65535||(n&65535)===65534||n>1114111?"�":String.fromCodePoint(n)}function hr(e){return e.replace(/[\t\n\r ]+/g," ").replace(/^ | $/g,"").toLowerCase().toUpperCase()}const It=zn(/[A-Za-z]/),rt=zn(/[\dA-Za-z]/),A_=zn(/[#-'*+\--9=?A-Z^-~]/);function wl(e){return e!==null&&(e<32||e===127)}const kl=zn(/\d/),R_=zn(/[\dA-Fa-f]/),I_=zn(/[!-/:-@[-`{-~]/);function X(e){return e!==null&&e<-2}function Ye(e){return e!==null&&(e<0||e===32)}function ie(e){return e===-2||e===-1||e===32}const j_=zn(new RegExp("\\p{P}|\\p{S}","u")),O_=zn(/\s/);function zn(e){return t;function t(n){return n!==null&&n>-1&&e.test(String.fromCharCode(n))}}function Tr(e){const t=[];let n=-1,r=0,i=0;for(;++n<e.length;){const a=e.charCodeAt(n);let o="";if(a===37&&rt(e.charCodeAt(n+1))&&rt(e.charCodeAt(n+2)))i=2;else if(a<128)/[!#$&-;=?-Z_a-z~]/.test(String.fromCharCode(a))||(o=String.fromCharCode(a));else if(a>55295&&a<57344){const s=e.charCodeAt(n+1);a<56320&&s>56319&&s<57344?(o=String.fromCharCode(a,s),i=1):o="�"}else o=String.fromCharCode(a);o&&(t.push(e.slice(r,n),encodeURIComponent(o)),r=n+i+1,o=""),i&&(n+=i,i=0)}return t.join("")+e.slice(r)}function de(e,t,n,r){const i=r?r-1:Number.POSITIVE_INFINITY;let a=0;return o;function o(l){return ie(l)?(e.enter(n),s(l)):t(l)}function s(l){return ie(l)&&a++<i?(e.consume(l),s):(e.exit(n),t(l))}}const M_={tokenize:D_};function D_(e){const t=e.attempt(this.parser.constructs.contentInitial,r,i);let n;return t;function r(s){if(s===null){e.consume(s);return}return e.enter("lineEnding"),e.consume(s),e.exit("lineEnding"),de(e,t,"linePrefix")}function i(s){return e.enter("paragraph"),a(s)}function a(s){const l=e.enter("chunkText",{contentType:"text",previous:n});return n&&(n.next=l),n=l,o(s)}function o(s){if(s===null){e.exit("chunkText"),e.exit("paragraph"),e.consume(s);return}return X(s)?(e.consume(s),e.exit("chunkText"),a):(e.consume(s),o)}}const F_={tokenize:W_},yd={tokenize:B_};function W_(e){const t=this,n=[];let r=0,i,a,o;return s;function s(g){if(r<n.length){const b=n[r];return t.containerState=b[1],e.attempt(b[0].continuation,l,u)(g)}return u(g)}function l(g){if(r++,t.containerState._closeFlow){t.containerState._closeFlow=void 0,i&&m();const b=t.events.length;let z=b,k;for(;z--;)if(t.events[z][0]==="exit"&&t.events[z][1].type==="chunkFlow"){k=t.events[z][1].end;break}h(r);let E=b;for(;E<t.events.length;)t.events[E][1].end={...k},E++;return Dt(t.events,z+1,0,t.events.slice(b)),t.events.length=E,u(g)}return s(g)}function u(g){if(r===n.length){if(!i)return p(g);if(i.currentConstruct&&i.currentConstruct.concrete)return w(g);t.interrupt=!!(i.currentConstruct&&!i._gfmTableDynamicInterruptHack)}return t.containerState={},e.check(yd,c,d)(g)}function c(g){return i&&m(),h(r),p(g)}function d(g){return t.parser.lazy[t.now().line]=r!==n.length,o=t.now().offset,w(g)}function p(g){return t.containerState={},e.attempt(yd,f,w)(g)}function f(g){return r++,n.push([t.currentConstruct,t.containerState]),p(g)}function w(g){if(g===null){i&&m(),h(0),e.consume(g);return}return i=i||t.parser.flow(t.now()),e.enter("chunkFlow",{_tokenizer:i,contentType:"flow",previous:a}),_(g)}function _(g){if(g===null){S(e.exit("chunkFlow"),!0),h(0),e.consume(g);return}return X(g)?(e.consume(g),S(e.exit("chunkFlow")),r=0,t.interrupt=void 0,s):(e.consume(g),_)}function S(g,b){const z=t.sliceStream(g);if(b&&z.push(null),g.previous=a,a&&(a.next=g),a=g,i.defineSkip(g.start),i.write(z),t.parser.lazy[g.start.line]){let k=i.events.length;for(;k--;)if(i.events[k][1].start.offset<o&&(!i.events[k][1].end||i.events[k][1].end.offset>o))return;const E=t.events.length;let N=E,O,L;for(;N--;)if(t.events[N][0]==="exit"&&t.events[N][1].type==="chunkFlow"){if(O){L=t.events[N][1].end;break}O=!0}for(h(r),k=E;k<t.events.length;)t.events[k][1].end={...L},k++;Dt(t.events,N+1,0,t.events.slice(E)),t.events.length=k}}function h(g){let b=n.length;for(;b-- >g;){const z=n[b];t.containerState=z[1],z[0].exit.call(t,e)}n.length=g}function m(){i.write([null]),a=void 0,i=void 0,t.containerState._closeFlow=void 0}}function B_(e,t,n){return de(e,e.attempt(this.parser.constructs.document,t,n),"linePrefix",this.parser.constructs.disable.null.includes("codeIndented")?void 0:4)}function xd(e){if(e===null||Ye(e)||O_(e))return 1;if(j_(e))return 2}function Tu(e,t,n){const r=[];let i=-1;for(;++i<e.length;){const a=e[i].resolveAll;a&&!r.includes(a)&&(t=a(t,n),r.push(a))}return t}const bl={name:"attention",resolveAll:q_,tokenize:U_};function q_(e,t){let n=-1,r,i,a,o,s,l,u,c;for(;++n<e.length;)if(e[n][0]==="enter"&&e[n][1].type==="attentionSequence"&&e[n][1]._close){for(r=n;r--;)if(e[r][0]==="exit"&&e[r][1].type==="attentionSequence"&&e[r][1]._open&&t.sliceSerialize(e[r][1]).charCodeAt(0)===t.sliceSerialize(e[n][1]).charCodeAt(0)){if((e[r][1]._close||e[n][1]._open)&&(e[n][1].end.offset-e[n][1].start.offset)%3&&!((e[r][1].end.offset-e[r][1].start.offset+e[n][1].end.offset-e[n][1].start.offset)%3))continue;l=e[r][1].end.offset-e[r][1].start.offset>1&&e[n][1].end.offset-e[n][1].start.offset>1?2:1;const d={...e[r][1].end},p={...e[n][1].start};_d(d,-l),_d(p,l),o={type:l>1?"strongSequence":"emphasisSequence",start:d,end:{...e[r][1].end}},s={type:l>1?"strongSequence":"emphasisSequence",start:{...e[n][1].start},end:p},a={type:l>1?"strongText":"emphasisText",start:{...e[r][1].end},end:{...e[n][1].start}},i={type:l>1?"strong":"emphasis",start:{...o.start},end:{...s.end}},e[r][1].end={...o.start},e[n][1].start={...s.end},u=[],e[r][1].end.offset-e[r][1].start.offset&&(u=pt(u,[["enter",e[r][1],t],["exit",e[r][1],t]])),u=pt(u,[["enter",i,t],["enter",o,t],["exit",o,t],["enter",a,t]]),u=pt(u,Tu(t.parser.constructs.insideSpan.null,e.slice(r+1,n),t)),u=pt(u,[["exit",a,t],["enter",s,t],["exit",s,t],["exit",i,t]]),e[n][1].end.offset-e[n][1].start.offset?(c=2,u=pt(u,[["enter",e[n][1],t],["exit",e[n][1],t]])):c=0,Dt(e,r-1,n-r+3,u),n=r+u.length-c-2;break}}for(n=-1;++n<e.length;)e[n][1].type==="attentionSequence"&&(e[n][1].type="data");return e}function U_(e,t){const n=this.parser.constructs.attentionMarkers.null,r=this.previous,i=xd(r);let a;return o;function o(l){return a=l,e.enter("attentionSequence"),s(l)}function s(l){if(l===a)return e.consume(l),s;const u=e.exit("attentionSequence"),c=xd(l),d=!c||c===2&&i||n.includes(l),p=!i||i===2&&c||n.includes(r);return u._open=!!(a===42?d:d&&(i||!p)),u._close=!!(a===42?p:p&&(c||!d)),t(l)}}function _d(e,t){e.column+=t,e.offset+=t,e._bufferIndex+=t}const V_={name:"autolink",tokenize:H_};function H_(e,t,n){let r=0;return i;function i(f){return e.enter("autolink"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.enter("autolinkProtocol"),a}function a(f){return It(f)?(e.consume(f),o):f===64?n(f):u(f)}function o(f){return f===43||f===45||f===46||rt(f)?(r=1,s(f)):u(f)}function s(f){return f===58?(e.consume(f),r=0,l):(f===43||f===45||f===46||rt(f))&&r++<32?(e.consume(f),s):(r=0,u(f))}function l(f){return f===62?(e.exit("autolinkProtocol"),e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):f===null||f===32||f===60||wl(f)?n(f):(e.consume(f),l)}function u(f){return f===64?(e.consume(f),c):A_(f)?(e.consume(f),u):n(f)}function c(f){return rt(f)?d(f):n(f)}function d(f){return f===46?(e.consume(f),r=0,c):f===62?(e.exit("autolinkProtocol").type="autolinkEmail",e.enter("autolinkMarker"),e.consume(f),e.exit("autolinkMarker"),e.exit("autolink"),t):p(f)}function p(f){if((f===45||rt(f))&&r++<63){const w=f===45?p:d;return e.consume(f),w}return n(f)}}const Co={partial:!0,tokenize:$_};function $_(e,t,n){return r;function r(a){return ie(a)?de(e,i,"linePrefix")(a):i(a)}function i(a){return a===null||X(a)?t(a):n(a)}}const Um={continuation:{tokenize:X_},exit:K_,name:"blockQuote",tokenize:Q_};function Q_(e,t,n){const r=this;return i;function i(o){if(o===62){const s=r.containerState;return s.open||(e.enter("blockQuote",{_container:!0}),s.open=!0),e.enter("blockQuotePrefix"),e.enter("blockQuoteMarker"),e.consume(o),e.exit("blockQuoteMarker"),a}return n(o)}function a(o){return ie(o)?(e.enter("blockQuotePrefixWhitespace"),e.consume(o),e.exit("blockQuotePrefixWhitespace"),e.exit("blockQuotePrefix"),t):(e.exit("blockQuotePrefix"),t(o))}}function X_(e,t,n){const r=this;return i;function i(o){return ie(o)?de(e,a,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(o):a(o)}function a(o){return e.attempt(Um,t,n)(o)}}function K_(e){e.exit("blockQuote")}const Vm={name:"characterEscape",tokenize:Y_};function Y_(e,t,n){return r;function r(a){return e.enter("characterEscape"),e.enter("escapeMarker"),e.consume(a),e.exit("escapeMarker"),i}function i(a){return I_(a)?(e.enter("characterEscapeValue"),e.consume(a),e.exit("characterEscapeValue"),e.exit("characterEscape"),t):n(a)}}const Hm={name:"characterReference",tokenize:G_};function G_(e,t,n){const r=this;let i=0,a,o;return s;function s(d){return e.enter("characterReference"),e.enter("characterReferenceMarker"),e.consume(d),e.exit("characterReferenceMarker"),l}function l(d){return d===35?(e.enter("characterReferenceMarkerNumeric"),e.consume(d),e.exit("characterReferenceMarkerNumeric"),u):(e.enter("characterReferenceValue"),a=31,o=rt,c(d))}function u(d){return d===88||d===120?(e.enter("characterReferenceMarkerHexadecimal"),e.consume(d),e.exit("characterReferenceMarkerHexadecimal"),e.enter("characterReferenceValue"),a=6,o=R_,c):(e.enter("characterReferenceValue"),a=7,o=kl,c(d))}function c(d){if(d===59&&i){const p=e.exit("characterReferenceValue");return o===rt&&!Nu(r.sliceSerialize(p))?n(d):(e.enter("characterReferenceMarker"),e.consume(d),e.exit("characterReferenceMarker"),e.exit("characterReference"),t)}return o(d)&&i++<a?(e.consume(d),c):n(d)}}const vd={partial:!0,tokenize:J_},wd={concrete:!0,name:"codeFenced",tokenize:Z_};function Z_(e,t,n){const r=this,i={partial:!0,tokenize:z};let a=0,o=0,s;return l;function l(k){return u(k)}function u(k){const E=r.events[r.events.length-1];return a=E&&E[1].type==="linePrefix"?E[2].sliceSerialize(E[1],!0).length:0,s=k,e.enter("codeFenced"),e.enter("codeFencedFence"),e.enter("codeFencedFenceSequence"),c(k)}function c(k){return k===s?(o++,e.consume(k),c):o<3?n(k):(e.exit("codeFencedFenceSequence"),ie(k)?de(e,d,"whitespace")(k):d(k))}function d(k){return k===null||X(k)?(e.exit("codeFencedFence"),r.interrupt?t(k):e.check(vd,_,b)(k)):(e.enter("codeFencedFenceInfo"),e.enter("chunkString",{contentType:"string"}),p(k))}function p(k){return k===null||X(k)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),d(k)):ie(k)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),de(e,f,"whitespace")(k)):k===96&&k===s?n(k):(e.consume(k),p)}function f(k){return k===null||X(k)?d(k):(e.enter("codeFencedFenceMeta"),e.enter("chunkString",{contentType:"string"}),w(k))}function w(k){return k===null||X(k)?(e.exit("chunkString"),e.exit("codeFencedFenceMeta"),d(k)):k===96&&k===s?n(k):(e.consume(k),w)}function _(k){return e.attempt(i,b,S)(k)}function S(k){return e.enter("lineEnding"),e.consume(k),e.exit("lineEnding"),h}function h(k){return a>0&&ie(k)?de(e,m,"linePrefix",a+1)(k):m(k)}function m(k){return k===null||X(k)?e.check(vd,_,b)(k):(e.enter("codeFlowValue"),g(k))}function g(k){return k===null||X(k)?(e.exit("codeFlowValue"),m(k)):(e.consume(k),g)}function b(k){return e.exit("codeFenced"),t(k)}function z(k,E,N){let O=0;return L;function L(j){return k.enter("lineEnding"),k.consume(j),k.exit("lineEnding"),F}function F(j){return k.enter("codeFencedFence"),ie(j)?de(k,U,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(j):U(j)}function U(j){return j===s?(k.enter("codeFencedFenceSequence"),Y(j)):N(j)}function Y(j){return j===s?(O++,k.consume(j),Y):O>=o?(k.exit("codeFencedFenceSequence"),ie(j)?de(k,re,"whitespace")(j):re(j)):N(j)}function re(j){return j===null||X(j)?(k.exit("codeFencedFence"),E(j)):N(j)}}}function J_(e,t,n){const r=this;return i;function i(o){return o===null?n(o):(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}const as={name:"codeIndented",tokenize:tv},ev={partial:!0,tokenize:nv};function tv(e,t,n){const r=this;return i;function i(u){return e.enter("codeIndented"),de(e,a,"linePrefix",5)(u)}function a(u){const c=r.events[r.events.length-1];return c&&c[1].type==="linePrefix"&&c[2].sliceSerialize(c[1],!0).length>=4?o(u):n(u)}function o(u){return u===null?l(u):X(u)?e.attempt(ev,o,l)(u):(e.enter("codeFlowValue"),s(u))}function s(u){return u===null||X(u)?(e.exit("codeFlowValue"),o(u)):(e.consume(u),s)}function l(u){return e.exit("codeIndented"),t(u)}}function nv(e,t,n){const r=this;return i;function i(o){return r.parser.lazy[r.now().line]?n(o):X(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),i):de(e,a,"linePrefix",5)(o)}function a(o){const s=r.events[r.events.length-1];return s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):X(o)?i(o):n(o)}}const rv={name:"codeText",previous:av,resolve:iv,tokenize:ov};function iv(e){let t=e.length-4,n=3,r,i;if((e[n][1].type==="lineEnding"||e[n][1].type==="space")&&(e[t][1].type==="lineEnding"||e[t][1].type==="space")){for(r=n;++r<t;)if(e[r][1].type==="codeTextData"){e[n][1].type="codeTextPadding",e[t][1].type="codeTextPadding",n+=2,t-=2;break}}for(r=n-1,t++;++r<=t;)i===void 0?r!==t&&e[r][1].type!=="lineEnding"&&(i=r):(r===t||e[r][1].type==="lineEnding")&&(e[i][1].type="codeTextData",r!==i+2&&(e[i][1].end=e[r-1][1].end,e.splice(i+2,r-i-2),t-=r-i-2,r=i+2),i=void 0);return e}function av(e){return e!==96||this.events[this.events.length-1][1].type==="characterEscape"}function ov(e,t,n){let r=0,i,a;return o;function o(d){return e.enter("codeText"),e.enter("codeTextSequence"),s(d)}function s(d){return d===96?(e.consume(d),r++,s):(e.exit("codeTextSequence"),l(d))}function l(d){return d===null?n(d):d===32?(e.enter("space"),e.consume(d),e.exit("space"),l):d===96?(a=e.enter("codeTextSequence"),i=0,c(d)):X(d)?(e.enter("lineEnding"),e.consume(d),e.exit("lineEnding"),l):(e.enter("codeTextData"),u(d))}function u(d){return d===null||d===32||d===96||X(d)?(e.exit("codeTextData"),l(d)):(e.consume(d),u)}function c(d){return d===96?(e.consume(d),i++,c):i===r?(e.exit("codeTextSequence"),e.exit("codeText"),t(d)):(a.type="codeTextData",u(d))}}class sv{constructor(t){this.left=t?[...t]:[],this.right=[]}get(t){if(t<0||t>=this.left.length+this.right.length)throw new RangeError("Cannot access index `"+t+"` in a splice buffer of size `"+(this.left.length+this.right.length)+"`");return t<this.left.length?this.left[t]:this.right[this.right.length-t+this.left.length-1]}get length(){return this.left.length+this.right.length}shift(){return this.setCursor(0),this.right.pop()}slice(t,n){const r=n??Number.POSITIVE_INFINITY;return r<this.left.length?this.left.slice(t,r):t>this.left.length?this.right.slice(this.right.length-r+this.left.length,this.right.length-t+this.left.length).reverse():this.left.slice(t).concat(this.right.slice(this.right.length-r+this.left.length).reverse())}splice(t,n,r){const i=n||0;this.setCursor(Math.trunc(t));const a=this.right.splice(this.right.length-i,Number.POSITIVE_INFINITY);return r&&Br(this.left,r),a.reverse()}pop(){return this.setCursor(Number.POSITIVE_INFINITY),this.left.pop()}push(t){this.setCursor(Number.POSITIVE_INFINITY),this.left.push(t)}pushMany(t){this.setCursor(Number.POSITIVE_INFINITY),Br(this.left,t)}unshift(t){this.setCursor(0),this.right.push(t)}unshiftMany(t){this.setCursor(0),Br(this.right,t.reverse())}setCursor(t){if(!(t===this.left.length||t>this.left.length&&this.right.length===0||t<0&&this.left.length===0))if(t<this.left.length){const n=this.left.splice(t,Number.POSITIVE_INFINITY);Br(this.right,n.reverse())}else{const n=this.right.splice(this.left.length+this.right.length-t,Number.POSITIVE_INFINITY);Br(this.left,n.reverse())}}}function Br(e,t){let n=0;if(t.length<1e4)e.push(...t);else for(;n<t.length;)e.push(...t.slice(n,n+1e4)),n+=1e4}function $m(e){const t={};let n=-1,r,i,a,o,s,l,u;const c=new sv(e);for(;++n<c.length;){for(;n in t;)n=t[n];if(r=c.get(n),n&&r[1].type==="chunkFlow"&&c.get(n-1)[1].type==="listItemPrefix"&&(l=r[1]._tokenizer.events,a=0,a<l.length&&l[a][1].type==="lineEndingBlank"&&(a+=2),a<l.length&&l[a][1].type==="content"))for(;++a<l.length&&l[a][1].type!=="content";)l[a][1].type==="chunkText"&&(l[a][1]._isInFirstContentOfListItem=!0,a++);if(r[0]==="enter")r[1].contentType&&(Object.assign(t,lv(c,n)),n=t[n],u=!0);else if(r[1]._container){for(a=n,i=void 0;a--;)if(o=c.get(a),o[1].type==="lineEnding"||o[1].type==="lineEndingBlank")o[0]==="enter"&&(i&&(c.get(i)[1].type="lineEndingBlank"),o[1].type="lineEnding",i=a);else if(!(o[1].type==="linePrefix"||o[1].type==="listItemIndent"))break;i&&(r[1].end={...c.get(i)[1].start},s=c.slice(i,n),s.unshift(r),c.splice(i,n-i+1,s))}}return Dt(e,0,Number.POSITIVE_INFINITY,c.slice(0)),!u}function lv(e,t){const n=e.get(t)[1],r=e.get(t)[2];let i=t-1;const a=[];let o=n._tokenizer;o||(o=r.parser[n.contentType](n.start),n._contentTypeTextTrailing&&(o._contentTypeTextTrailing=!0));const s=o.events,l=[],u={};let c,d,p=-1,f=n,w=0,_=0;const S=[_];for(;f;){for(;e.get(++i)[1]!==f;);a.push(i),f._tokenizer||(c=r.sliceStream(f),f.next||c.push(null),d&&o.defineSkip(f.start),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=!0),o.write(c),f._isInFirstContentOfListItem&&(o._gfmTasklistFirstContentOfListItem=void 0)),d=f,f=f.next}for(f=n;++p<s.length;)s[p][0]==="exit"&&s[p-1][0]==="enter"&&s[p][1].type===s[p-1][1].type&&s[p][1].start.line!==s[p][1].end.line&&(_=p+1,S.push(_),f._tokenizer=void 0,f.previous=void 0,f=f.next);for(o.events=[],f?(f._tokenizer=void 0,f.previous=void 0):S.pop(),p=S.length;p--;){const h=s.slice(S[p],S[p+1]),m=a.pop();l.push([m,m+h.length-1]),e.splice(m,2,h)}for(l.reverse(),p=-1;++p<l.length;)u[w+l[p][0]]=w+l[p][1],w+=l[p][1]-l[p][0]-1;return u}const uv={resolve:dv,tokenize:pv},cv={partial:!0,tokenize:fv};function dv(e){return $m(e),e}function pv(e,t){let n;return r;function r(s){return e.enter("content"),n=e.enter("chunkContent",{contentType:"content"}),i(s)}function i(s){return s===null?a(s):X(s)?e.check(cv,o,a)(s):(e.consume(s),i)}function a(s){return e.exit("chunkContent"),e.exit("content"),t(s)}function o(s){return e.consume(s),e.exit("chunkContent"),n.next=e.enter("chunkContent",{contentType:"content",previous:n}),n=n.next,i}}function fv(e,t,n){const r=this;return i;function i(o){return e.exit("chunkContent"),e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),de(e,a,"linePrefix")}function a(o){if(o===null||X(o))return n(o);const s=r.events[r.events.length-1];return!r.parser.constructs.disable.null.includes("codeIndented")&&s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(o):e.interrupt(r.parser.constructs.flow,n,t)(o)}}function Qm(e,t,n,r,i,a,o,s,l){const u=l||Number.POSITIVE_INFINITY;let c=0;return d;function d(h){return h===60?(e.enter(r),e.enter(i),e.enter(a),e.consume(h),e.exit(a),p):h===null||h===32||h===41||wl(h)?n(h):(e.enter(r),e.enter(o),e.enter(s),e.enter("chunkString",{contentType:"string"}),_(h))}function p(h){return h===62?(e.enter(a),e.consume(h),e.exit(a),e.exit(i),e.exit(r),t):(e.enter(s),e.enter("chunkString",{contentType:"string"}),f(h))}function f(h){return h===62?(e.exit("chunkString"),e.exit(s),p(h)):h===null||h===60||X(h)?n(h):(e.consume(h),h===92?w:f)}function w(h){return h===60||h===62||h===92?(e.consume(h),f):f(h)}function _(h){return!c&&(h===null||h===41||Ye(h))?(e.exit("chunkString"),e.exit(s),e.exit(o),e.exit(r),t(h)):c<u&&h===40?(e.consume(h),c++,_):h===41?(e.consume(h),c--,_):h===null||h===32||h===40||wl(h)?n(h):(e.consume(h),h===92?S:_)}function S(h){return h===40||h===41||h===92?(e.consume(h),_):_(h)}}function Xm(e,t,n,r,i,a){const o=this;let s=0,l;return u;function u(f){return e.enter(r),e.enter(i),e.consume(f),e.exit(i),e.enter(a),c}function c(f){return s>999||f===null||f===91||f===93&&!l||f===94&&!s&&"_hiddenFootnoteSupport"in o.parser.constructs?n(f):f===93?(e.exit(a),e.enter(i),e.consume(f),e.exit(i),e.exit(r),t):X(f)?(e.enter("lineEnding"),e.consume(f),e.exit("lineEnding"),c):(e.enter("chunkString",{contentType:"string"}),d(f))}function d(f){return f===null||f===91||f===93||X(f)||s++>999?(e.exit("chunkString"),c(f)):(e.consume(f),l||(l=!ie(f)),f===92?p:d)}function p(f){return f===91||f===92||f===93?(e.consume(f),s++,d):d(f)}}function Km(e,t,n,r,i,a){let o;return s;function s(p){return p===34||p===39||p===40?(e.enter(r),e.enter(i),e.consume(p),e.exit(i),o=p===40?41:p,l):n(p)}function l(p){return p===o?(e.enter(i),e.consume(p),e.exit(i),e.exit(r),t):(e.enter(a),u(p))}function u(p){return p===o?(e.exit(a),l(o)):p===null?n(p):X(p)?(e.enter("lineEnding"),e.consume(p),e.exit("lineEnding"),de(e,u,"linePrefix")):(e.enter("chunkString",{contentType:"string"}),c(p))}function c(p){return p===o||p===null||X(p)?(e.exit("chunkString"),u(p)):(e.consume(p),p===92?d:c)}function d(p){return p===o||p===92?(e.consume(p),c):c(p)}}function ai(e,t){let n;return r;function r(i){return X(i)?(e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),n=!0,r):ie(i)?de(e,r,n?"linePrefix":"lineSuffix")(i):t(i)}}const mv={name:"definition",tokenize:gv},hv={partial:!0,tokenize:yv};function gv(e,t,n){const r=this;let i;return a;function a(f){return e.enter("definition"),o(f)}function o(f){return Xm.call(r,e,s,n,"definitionLabel","definitionLabelMarker","definitionLabelString")(f)}function s(f){return i=hr(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)),f===58?(e.enter("definitionMarker"),e.consume(f),e.exit("definitionMarker"),l):n(f)}function l(f){return Ye(f)?ai(e,u)(f):u(f)}function u(f){return Qm(e,c,n,"definitionDestination","definitionDestinationLiteral","definitionDestinationLiteralMarker","definitionDestinationRaw","definitionDestinationString")(f)}function c(f){return e.attempt(hv,d,d)(f)}function d(f){return ie(f)?de(e,p,"whitespace")(f):p(f)}function p(f){return f===null||X(f)?(e.exit("definition"),r.parser.defined.push(i),t(f)):n(f)}}function yv(e,t,n){return r;function r(s){return Ye(s)?ai(e,i)(s):n(s)}function i(s){return Km(e,a,n,"definitionTitle","definitionTitleMarker","definitionTitleString")(s)}function a(s){return ie(s)?de(e,o,"whitespace")(s):o(s)}function o(s){return s===null||X(s)?t(s):n(s)}}const xv={name:"hardBreakEscape",tokenize:_v};function _v(e,t,n){return r;function r(a){return e.enter("hardBreakEscape"),e.consume(a),i}function i(a){return X(a)?(e.exit("hardBreakEscape"),t(a)):n(a)}}const vv={name:"headingAtx",resolve:wv,tokenize:kv};function wv(e,t){let n=e.length-2,r=3,i,a;return e[r][1].type==="whitespace"&&(r+=2),n-2>r&&e[n][1].type==="whitespace"&&(n-=2),e[n][1].type==="atxHeadingSequence"&&(r===n-1||n-4>r&&e[n-2][1].type==="whitespace")&&(n-=r+1===n?2:4),n>r&&(i={type:"atxHeadingText",start:e[r][1].start,end:e[n][1].end},a={type:"chunkText",start:e[r][1].start,end:e[n][1].end,contentType:"text"},Dt(e,r,n-r+1,[["enter",i,t],["enter",a,t],["exit",a,t],["exit",i,t]])),e}function kv(e,t,n){let r=0;return i;function i(c){return e.enter("atxHeading"),a(c)}function a(c){return e.enter("atxHeadingSequence"),o(c)}function o(c){return c===35&&r++<6?(e.consume(c),o):c===null||Ye(c)?(e.exit("atxHeadingSequence"),s(c)):n(c)}function s(c){return c===35?(e.enter("atxHeadingSequence"),l(c)):c===null||X(c)?(e.exit("atxHeading"),t(c)):ie(c)?de(e,s,"whitespace")(c):(e.enter("atxHeadingText"),u(c))}function l(c){return c===35?(e.consume(c),l):(e.exit("atxHeadingSequence"),s(c))}function u(c){return c===null||c===35||Ye(c)?(e.exit("atxHeadingText"),s(c)):(e.consume(c),u)}}const bv=["address","article","aside","base","basefont","blockquote","body","caption","center","col","colgroup","dd","details","dialog","dir","div","dl","dt","fieldset","figcaption","figure","footer","form","frame","frameset","h1","h2","h3","h4","h5","h6","head","header","hr","html","iframe","legend","li","link","main","menu","menuitem","nav","noframes","ol","optgroup","option","p","param","search","section","summary","table","tbody","td","tfoot","th","thead","title","tr","track","ul"],kd=["pre","script","style","textarea"],Sv={concrete:!0,name:"htmlFlow",resolveTo:Ev,tokenize:Pv},Cv={partial:!0,tokenize:Tv},zv={partial:!0,tokenize:Nv};function Ev(e){let t=e.length;for(;t--&&!(e[t][0]==="enter"&&e[t][1].type==="htmlFlow"););return t>1&&e[t-2][1].type==="linePrefix"&&(e[t][1].start=e[t-2][1].start,e[t+1][1].start=e[t-2][1].start,e.splice(t-2,2)),e}function Pv(e,t,n){const r=this;let i,a,o,s,l;return u;function u(v){return c(v)}function c(v){return e.enter("htmlFlow"),e.enter("htmlFlowData"),e.consume(v),d}function d(v){return v===33?(e.consume(v),p):v===47?(e.consume(v),a=!0,_):v===63?(e.consume(v),i=3,r.interrupt?t:y):It(v)?(e.consume(v),o=String.fromCharCode(v),S):n(v)}function p(v){return v===45?(e.consume(v),i=2,f):v===91?(e.consume(v),i=5,s=0,w):It(v)?(e.consume(v),i=4,r.interrupt?t:y):n(v)}function f(v){return v===45?(e.consume(v),r.interrupt?t:y):n(v)}function w(v){const te="CDATA[";return v===te.charCodeAt(s++)?(e.consume(v),s===te.length?r.interrupt?t:U:w):n(v)}function _(v){return It(v)?(e.consume(v),o=String.fromCharCode(v),S):n(v)}function S(v){if(v===null||v===47||v===62||Ye(v)){const te=v===47,we=o.toLowerCase();return!te&&!a&&kd.includes(we)?(i=1,r.interrupt?t(v):U(v)):bv.includes(o.toLowerCase())?(i=6,te?(e.consume(v),h):r.interrupt?t(v):U(v)):(i=7,r.interrupt&&!r.parser.lazy[r.now().line]?n(v):a?m(v):g(v))}return v===45||rt(v)?(e.consume(v),o+=String.fromCharCode(v),S):n(v)}function h(v){return v===62?(e.consume(v),r.interrupt?t:U):n(v)}function m(v){return ie(v)?(e.consume(v),m):L(v)}function g(v){return v===47?(e.consume(v),L):v===58||v===95||It(v)?(e.consume(v),b):ie(v)?(e.consume(v),g):L(v)}function b(v){return v===45||v===46||v===58||v===95||rt(v)?(e.consume(v),b):z(v)}function z(v){return v===61?(e.consume(v),k):ie(v)?(e.consume(v),z):g(v)}function k(v){return v===null||v===60||v===61||v===62||v===96?n(v):v===34||v===39?(e.consume(v),l=v,E):ie(v)?(e.consume(v),k):N(v)}function E(v){return v===l?(e.consume(v),l=null,O):v===null||X(v)?n(v):(e.consume(v),E)}function N(v){return v===null||v===34||v===39||v===47||v===60||v===61||v===62||v===96||Ye(v)?z(v):(e.consume(v),N)}function O(v){return v===47||v===62||ie(v)?g(v):n(v)}function L(v){return v===62?(e.consume(v),F):n(v)}function F(v){return v===null||X(v)?U(v):ie(v)?(e.consume(v),F):n(v)}function U(v){return v===45&&i===2?(e.consume(v),Q):v===60&&i===1?(e.consume(v),ee):v===62&&i===4?(e.consume(v),B):v===63&&i===3?(e.consume(v),y):v===93&&i===5?(e.consume(v),V):X(v)&&(i===6||i===7)?(e.exit("htmlFlowData"),e.check(Cv,$,Y)(v)):v===null||X(v)?(e.exit("htmlFlowData"),Y(v)):(e.consume(v),U)}function Y(v){return e.check(zv,re,$)(v)}function re(v){return e.enter("lineEnding"),e.consume(v),e.exit("lineEnding"),j}function j(v){return v===null||X(v)?Y(v):(e.enter("htmlFlowData"),U(v))}function Q(v){return v===45?(e.consume(v),y):U(v)}function ee(v){return v===47?(e.consume(v),o="",I):U(v)}function I(v){if(v===62){const te=o.toLowerCase();return kd.includes(te)?(e.consume(v),B):U(v)}return It(v)&&o.length<8?(e.consume(v),o+=String.fromCharCode(v),I):U(v)}function V(v){return v===93?(e.consume(v),y):U(v)}function y(v){return v===62?(e.consume(v),B):v===45&&i===2?(e.consume(v),y):U(v)}function B(v){return v===null||X(v)?(e.exit("htmlFlowData"),$(v)):(e.consume(v),B)}function $(v){return e.exit("htmlFlow"),t(v)}}function Nv(e,t,n){const r=this;return i;function i(o){return X(o)?(e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),a):n(o)}function a(o){return r.parser.lazy[r.now().line]?n(o):t(o)}}function Tv(e,t,n){return r;function r(i){return e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),e.attempt(Co,t,n)}}const Lv={name:"htmlText",tokenize:Av};function Av(e,t,n){const r=this;let i,a,o;return s;function s(y){return e.enter("htmlText"),e.enter("htmlTextData"),e.consume(y),l}function l(y){return y===33?(e.consume(y),u):y===47?(e.consume(y),z):y===63?(e.consume(y),g):It(y)?(e.consume(y),N):n(y)}function u(y){return y===45?(e.consume(y),c):y===91?(e.consume(y),a=0,w):It(y)?(e.consume(y),m):n(y)}function c(y){return y===45?(e.consume(y),f):n(y)}function d(y){return y===null?n(y):y===45?(e.consume(y),p):X(y)?(o=d,ee(y)):(e.consume(y),d)}function p(y){return y===45?(e.consume(y),f):d(y)}function f(y){return y===62?Q(y):y===45?p(y):d(y)}function w(y){const B="CDATA[";return y===B.charCodeAt(a++)?(e.consume(y),a===B.length?_:w):n(y)}function _(y){return y===null?n(y):y===93?(e.consume(y),S):X(y)?(o=_,ee(y)):(e.consume(y),_)}function S(y){return y===93?(e.consume(y),h):_(y)}function h(y){return y===62?Q(y):y===93?(e.consume(y),h):_(y)}function m(y){return y===null||y===62?Q(y):X(y)?(o=m,ee(y)):(e.consume(y),m)}function g(y){return y===null?n(y):y===63?(e.consume(y),b):X(y)?(o=g,ee(y)):(e.consume(y),g)}function b(y){return y===62?Q(y):g(y)}function z(y){return It(y)?(e.consume(y),k):n(y)}function k(y){return y===45||rt(y)?(e.consume(y),k):E(y)}function E(y){return X(y)?(o=E,ee(y)):ie(y)?(e.consume(y),E):Q(y)}function N(y){return y===45||rt(y)?(e.consume(y),N):y===47||y===62||Ye(y)?O(y):n(y)}function O(y){return y===47?(e.consume(y),Q):y===58||y===95||It(y)?(e.consume(y),L):X(y)?(o=O,ee(y)):ie(y)?(e.consume(y),O):Q(y)}function L(y){return y===45||y===46||y===58||y===95||rt(y)?(e.consume(y),L):F(y)}function F(y){return y===61?(e.consume(y),U):X(y)?(o=F,ee(y)):ie(y)?(e.consume(y),F):O(y)}function U(y){return y===null||y===60||y===61||y===62||y===96?n(y):y===34||y===39?(e.consume(y),i=y,Y):X(y)?(o=U,ee(y)):ie(y)?(e.consume(y),U):(e.consume(y),re)}function Y(y){return y===i?(e.consume(y),i=void 0,j):y===null?n(y):X(y)?(o=Y,ee(y)):(e.consume(y),Y)}function re(y){return y===null||y===34||y===39||y===60||y===61||y===96?n(y):y===47||y===62||Ye(y)?O(y):(e.consume(y),re)}function j(y){return y===47||y===62||Ye(y)?O(y):n(y)}function Q(y){return y===62?(e.consume(y),e.exit("htmlTextData"),e.exit("htmlText"),t):n(y)}function ee(y){return e.exit("htmlTextData"),e.enter("lineEnding"),e.consume(y),e.exit("lineEnding"),I}function I(y){return ie(y)?de(e,V,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(y):V(y)}function V(y){return e.enter("htmlTextData"),o(y)}}const Lu={name:"labelEnd",resolveAll:Ov,resolveTo:Mv,tokenize:Dv},Rv={tokenize:Fv},Iv={tokenize:Wv},jv={tokenize:Bv};function Ov(e){let t=-1;const n=[];for(;++t<e.length;){const r=e[t][1];if(n.push(e[t]),r.type==="labelImage"||r.type==="labelLink"||r.type==="labelEnd"){const i=r.type==="labelImage"?4:2;r.type="data",t+=i}}return e.length!==n.length&&Dt(e,0,e.length,n),e}function Mv(e,t){let n=e.length,r=0,i,a,o,s;for(;n--;)if(i=e[n][1],a){if(i.type==="link"||i.type==="labelLink"&&i._inactive)break;e[n][0]==="enter"&&i.type==="labelLink"&&(i._inactive=!0)}else if(o){if(e[n][0]==="enter"&&(i.type==="labelImage"||i.type==="labelLink")&&!i._balanced&&(a=n,i.type!=="labelLink")){r=2;break}}else i.type==="labelEnd"&&(o=n);const l={type:e[a][1].type==="labelLink"?"link":"image",start:{...e[a][1].start},end:{...e[e.length-1][1].end}},u={type:"label",start:{...e[a][1].start},end:{...e[o][1].end}},c={type:"labelText",start:{...e[a+r+2][1].end},end:{...e[o-2][1].start}};return s=[["enter",l,t],["enter",u,t]],s=pt(s,e.slice(a+1,a+r+3)),s=pt(s,[["enter",c,t]]),s=pt(s,Tu(t.parser.constructs.insideSpan.null,e.slice(a+r+4,o-3),t)),s=pt(s,[["exit",c,t],e[o-2],e[o-1],["exit",u,t]]),s=pt(s,e.slice(o+1)),s=pt(s,[["exit",l,t]]),Dt(e,a,e.length,s),e}function Dv(e,t,n){const r=this;let i=r.events.length,a,o;for(;i--;)if((r.events[i][1].type==="labelImage"||r.events[i][1].type==="labelLink")&&!r.events[i][1]._balanced){a=r.events[i][1];break}return s;function s(p){return a?a._inactive?d(p):(o=r.parser.defined.includes(hr(r.sliceSerialize({start:a.end,end:r.now()}))),e.enter("labelEnd"),e.enter("labelMarker"),e.consume(p),e.exit("labelMarker"),e.exit("labelEnd"),l):n(p)}function l(p){return p===40?e.attempt(Rv,c,o?c:d)(p):p===91?e.attempt(Iv,c,o?u:d)(p):o?c(p):d(p)}function u(p){return e.attempt(jv,c,d)(p)}function c(p){return t(p)}function d(p){return a._balanced=!0,n(p)}}function Fv(e,t,n){return r;function r(d){return e.enter("resource"),e.enter("resourceMarker"),e.consume(d),e.exit("resourceMarker"),i}function i(d){return Ye(d)?ai(e,a)(d):a(d)}function a(d){return d===41?c(d):Qm(e,o,s,"resourceDestination","resourceDestinationLiteral","resourceDestinationLiteralMarker","resourceDestinationRaw","resourceDestinationString",32)(d)}function o(d){return Ye(d)?ai(e,l)(d):c(d)}function s(d){return n(d)}function l(d){return d===34||d===39||d===40?Km(e,u,n,"resourceTitle","resourceTitleMarker","resourceTitleString")(d):c(d)}function u(d){return Ye(d)?ai(e,c)(d):c(d)}function c(d){return d===41?(e.enter("resourceMarker"),e.consume(d),e.exit("resourceMarker"),e.exit("resource"),t):n(d)}}function Wv(e,t,n){const r=this;return i;function i(s){return Xm.call(r,e,a,o,"reference","referenceMarker","referenceString")(s)}function a(s){return r.parser.defined.includes(hr(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)))?t(s):n(s)}function o(s){return n(s)}}function Bv(e,t,n){return r;function r(a){return e.enter("reference"),e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),i}function i(a){return a===93?(e.enter("referenceMarker"),e.consume(a),e.exit("referenceMarker"),e.exit("reference"),t):n(a)}}const qv={name:"labelStartImage",resolveAll:Lu.resolveAll,tokenize:Uv};function Uv(e,t,n){const r=this;return i;function i(s){return e.enter("labelImage"),e.enter("labelImageMarker"),e.consume(s),e.exit("labelImageMarker"),a}function a(s){return s===91?(e.enter("labelMarker"),e.consume(s),e.exit("labelMarker"),e.exit("labelImage"),o):n(s)}function o(s){return s===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(s):t(s)}}const Vv={name:"labelStartLink",resolveAll:Lu.resolveAll,tokenize:Hv};function Hv(e,t,n){const r=this;return i;function i(o){return e.enter("labelLink"),e.enter("labelMarker"),e.consume(o),e.exit("labelMarker"),e.exit("labelLink"),a}function a(o){return o===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(o):t(o)}}const os={name:"lineEnding",tokenize:$v};function $v(e,t){return n;function n(r){return e.enter("lineEnding"),e.consume(r),e.exit("lineEnding"),de(e,t,"linePrefix")}}const Sa={name:"thematicBreak",tokenize:Qv};function Qv(e,t,n){let r=0,i;return a;function a(u){return e.enter("thematicBreak"),o(u)}function o(u){return i=u,s(u)}function s(u){return u===i?(e.enter("thematicBreakSequence"),l(u)):r>=3&&(u===null||X(u))?(e.exit("thematicBreak"),t(u)):n(u)}function l(u){return u===i?(e.consume(u),r++,l):(e.exit("thematicBreakSequence"),ie(u)?de(e,s,"whitespace")(u):s(u))}}const Ve={continuation:{tokenize:Gv},exit:Jv,name:"list",tokenize:Yv},Xv={partial:!0,tokenize:e1},Kv={partial:!0,tokenize:Zv};function Yv(e,t,n){const r=this,i=r.events[r.events.length-1];let a=i&&i[1].type==="linePrefix"?i[2].sliceSerialize(i[1],!0).length:0,o=0;return s;function s(f){const w=r.containerState.type||(f===42||f===43||f===45?"listUnordered":"listOrdered");if(w==="listUnordered"?!r.containerState.marker||f===r.containerState.marker:kl(f)){if(r.containerState.type||(r.containerState.type=w,e.enter(w,{_container:!0})),w==="listUnordered")return e.enter("listItemPrefix"),f===42||f===45?e.check(Sa,n,u)(f):u(f);if(!r.interrupt||f===49)return e.enter("listItemPrefix"),e.enter("listItemValue"),l(f)}return n(f)}function l(f){return kl(f)&&++o<10?(e.consume(f),l):(!r.interrupt||o<2)&&(r.containerState.marker?f===r.containerState.marker:f===41||f===46)?(e.exit("listItemValue"),u(f)):n(f)}function u(f){return e.enter("listItemMarker"),e.consume(f),e.exit("listItemMarker"),r.containerState.marker=r.containerState.marker||f,e.check(Co,r.interrupt?n:c,e.attempt(Xv,p,d))}function c(f){return r.containerState.initialBlankLine=!0,a++,p(f)}function d(f){return ie(f)?(e.enter("listItemPrefixWhitespace"),e.consume(f),e.exit("listItemPrefixWhitespace"),p):n(f)}function p(f){return r.containerState.size=a+r.sliceSerialize(e.exit("listItemPrefix"),!0).length,t(f)}}function Gv(e,t,n){const r=this;return r.containerState._closeFlow=void 0,e.check(Co,i,a);function i(s){return r.containerState.furtherBlankLines=r.containerState.furtherBlankLines||r.containerState.initialBlankLine,de(e,t,"listItemIndent",r.containerState.size+1)(s)}function a(s){return r.containerState.furtherBlankLines||!ie(s)?(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,o(s)):(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,e.attempt(Kv,t,o)(s))}function o(s){return r.containerState._closeFlow=!0,r.interrupt=void 0,de(e,e.attempt(Ve,t,n),"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(s)}}function Zv(e,t,n){const r=this;return de(e,i,"listItemIndent",r.containerState.size+1);function i(a){const o=r.events[r.events.length-1];return o&&o[1].type==="listItemIndent"&&o[2].sliceSerialize(o[1],!0).length===r.containerState.size?t(a):n(a)}}function Jv(e){e.exit(this.containerState.type)}function e1(e,t,n){const r=this;return de(e,i,"listItemPrefixWhitespace",r.parser.constructs.disable.null.includes("codeIndented")?void 0:5);function i(a){const o=r.events[r.events.length-1];return!ie(a)&&o&&o[1].type==="listItemPrefixWhitespace"?t(a):n(a)}}const bd={name:"setextUnderline",resolveTo:t1,tokenize:n1};function t1(e,t){let n=e.length,r,i,a;for(;n--;)if(e[n][0]==="enter"){if(e[n][1].type==="content"){r=n;break}e[n][1].type==="paragraph"&&(i=n)}else e[n][1].type==="content"&&e.splice(n,1),!a&&e[n][1].type==="definition"&&(a=n);const o={type:"setextHeading",start:{...e[r][1].start},end:{...e[e.length-1][1].end}};return e[i][1].type="setextHeadingText",a?(e.splice(i,0,["enter",o,t]),e.splice(a+1,0,["exit",e[r][1],t]),e[r][1].end={...e[a][1].end}):e[r][1]=o,e.push(["exit",o,t]),e}function n1(e,t,n){const r=this;let i;return a;function a(u){let c=r.events.length,d;for(;c--;)if(r.events[c][1].type!=="lineEnding"&&r.events[c][1].type!=="linePrefix"&&r.events[c][1].type!=="content"){d=r.events[c][1].type==="paragraph";break}return!r.parser.lazy[r.now().line]&&(r.interrupt||d)?(e.enter("setextHeadingLine"),i=u,o(u)):n(u)}function o(u){return e.enter("setextHeadingLineSequence"),s(u)}function s(u){return u===i?(e.consume(u),s):(e.exit("setextHeadingLineSequence"),ie(u)?de(e,l,"lineSuffix")(u):l(u))}function l(u){return u===null||X(u)?(e.exit("setextHeadingLine"),t(u)):n(u)}}const r1={tokenize:i1};function i1(e){const t=this,n=e.attempt(Co,r,e.attempt(this.parser.constructs.flowInitial,i,de(e,e.attempt(this.parser.constructs.flow,i,e.attempt(uv,i)),"linePrefix")));return n;function r(a){if(a===null){e.consume(a);return}return e.enter("lineEndingBlank"),e.consume(a),e.exit("lineEndingBlank"),t.currentConstruct=void 0,n}function i(a){if(a===null){e.consume(a);return}return e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),t.currentConstruct=void 0,n}}const a1={resolveAll:Gm()},o1=Ym("string"),s1=Ym("text");function Ym(e){return{resolveAll:Gm(e==="text"?l1:void 0),tokenize:t};function t(n){const r=this,i=this.parser.constructs[e],a=n.attempt(i,o,s);return o;function o(c){return u(c)?a(c):s(c)}function s(c){if(c===null){n.consume(c);return}return n.enter("data"),n.consume(c),l}function l(c){return u(c)?(n.exit("data"),a(c)):(n.consume(c),l)}function u(c){if(c===null)return!0;const d=i[c];let p=-1;if(d)for(;++p<d.length;){const f=d[p];if(!f.previous||f.previous.call(r,r.previous))return!0}return!1}}}function Gm(e){return t;function t(n,r){let i=-1,a;for(;++i<=n.length;)a===void 0?n[i]&&n[i][1].type==="data"&&(a=i,i++):(!n[i]||n[i][1].type!=="data")&&(i!==a+2&&(n[a][1].end=n[i-1][1].end,n.splice(a+2,i-a-2),i=a+2),a=void 0);return e?e(n,r):n}}function l1(e,t){let n=0;for(;++n<=e.length;)if((n===e.length||e[n][1].type==="lineEnding")&&e[n-1][1].type==="data"){const r=e[n-1][1],i=t.sliceStream(r);let a=i.length,o=-1,s=0,l;for(;a--;){const u=i[a];if(typeof u=="string"){for(o=u.length;u.charCodeAt(o-1)===32;)s++,o--;if(o)break;o=-1}else if(u===-2)l=!0,s++;else if(u!==-1){a++;break}}if(t._contentTypeTextTrailing&&n===e.length&&(s=0),s){const u={type:n===e.length||l||s<2?"lineSuffix":"hardBreakTrailing",start:{_bufferIndex:a?o:r.start._bufferIndex+o,_index:r.start._index+a,line:r.end.line,column:r.end.column-s,offset:r.end.offset-s},end:{...r.end}};r.end={...u.start},r.start.offset===r.end.offset?Object.assign(r,u):(e.splice(n,0,["enter",u,t],["exit",u,t]),n+=2)}n++}return e}const u1={42:Ve,43:Ve,45:Ve,48:Ve,49:Ve,50:Ve,51:Ve,52:Ve,53:Ve,54:Ve,55:Ve,56:Ve,57:Ve,62:Um},c1={91:mv},d1={[-2]:as,[-1]:as,32:as},p1={35:vv,42:Sa,45:[bd,Sa],60:Sv,61:bd,95:Sa,96:wd,126:wd},f1={38:Hm,92:Vm},m1={[-5]:os,[-4]:os,[-3]:os,33:qv,38:Hm,42:bl,60:[V_,Lv],91:Vv,92:[xv,Vm],93:Lu,95:bl,96:rv},h1={null:[bl,a1]},g1={null:[42,95]},y1={null:[]},x1=Object.freeze(Object.defineProperty({__proto__:null,attentionMarkers:g1,contentInitial:c1,disable:y1,document:u1,flow:p1,flowInitial:d1,insideSpan:h1,string:f1,text:m1},Symbol.toStringTag,{value:"Module"}));function _1(e,t,n){let r={_bufferIndex:-1,_index:0,line:n&&n.line||1,column:n&&n.column||1,offset:n&&n.offset||0};const i={},a=[];let o=[],s=[];const l={attempt:E(z),check:E(k),consume:m,enter:g,exit:b,interrupt:E(k,{interrupt:!0})},u={code:null,containerState:{},defineSkip:_,events:[],now:w,parser:e,previous:null,sliceSerialize:p,sliceStream:f,write:d};let c=t.tokenize.call(u,l);return t.resolveAll&&a.push(t),u;function d(F){return o=pt(o,F),S(),o[o.length-1]!==null?[]:(N(t,0),u.events=Tu(a,u.events,u),u.events)}function p(F,U){return w1(f(F),U)}function f(F){return v1(o,F)}function w(){const{_bufferIndex:F,_index:U,line:Y,column:re,offset:j}=r;return{_bufferIndex:F,_index:U,line:Y,column:re,offset:j}}function _(F){i[F.line]=F.column,L()}function S(){let F;for(;r._index<o.length;){const U=o[r._index];if(typeof U=="string")for(F=r._index,r._bufferIndex<0&&(r._bufferIndex=0);r._index===F&&r._bufferIndex<U.length;)h(U.charCodeAt(r._bufferIndex));else h(U)}}function h(F){c=c(F)}function m(F){X(F)?(r.line++,r.column=1,r.offset+=F===-3?2:1,L()):F!==-1&&(r.column++,r.offset++),r._bufferIndex<0?r._index++:(r._bufferIndex++,r._bufferIndex===o[r._index].length&&(r._bufferIndex=-1,r._index++)),u.previous=F}function g(F,U){const Y=U||{};return Y.type=F,Y.start=w(),u.events.push(["enter",Y,u]),s.push(Y),Y}function b(F){const U=s.pop();return U.end=w(),u.events.push(["exit",U,u]),U}function z(F,U){N(F,U.from)}function k(F,U){U.restore()}function E(F,U){return Y;function Y(re,j,Q){let ee,I,V,y;return Array.isArray(re)?$(re):"tokenize"in re?$([re]):B(re);function B(se){return lt;function lt(W){const T=W!==null&&se[W],M=W!==null&&se.null,H=[...Array.isArray(T)?T:T?[T]:[],...Array.isArray(M)?M:M?[M]:[]];return $(H)(W)}}function $(se){return ee=se,I=0,se.length===0?Q:v(se[I])}function v(se){return lt;function lt(W){return y=O(),V=se,se.partial||(u.currentConstruct=se),se.name&&u.parser.constructs.disable.null.includes(se.name)?we():se.tokenize.call(U?Object.assign(Object.create(u),U):u,l,te,we)(W)}}function te(se){return F(V,y),j}function we(se){return y.restore(),++I<ee.length?v(ee[I]):Q}}}function N(F,U){F.resolveAll&&!a.includes(F)&&a.push(F),F.resolve&&Dt(u.events,U,u.events.length-U,F.resolve(u.events.slice(U),u)),F.resolveTo&&(u.events=F.resolveTo(u.events,u))}function O(){const F=w(),U=u.previous,Y=u.currentConstruct,re=u.events.length,j=Array.from(s);return{from:re,restore:Q};function Q(){r=F,u.previous=U,u.currentConstruct=Y,u.events.length=re,s=j,L()}}function L(){r.line in i&&r.column<2&&(r.column=i[r.line],r.offset+=i[r.line]-1)}}function v1(e,t){const n=t.start._index,r=t.start._bufferIndex,i=t.end._index,a=t.end._bufferIndex;let o;if(n===i)o=[e[n].slice(r,a)];else{if(o=e.slice(n,i),r>-1){const s=o[0];typeof s=="string"?o[0]=s.slice(r):o.shift()}a>0&&o.push(e[i].slice(0,a))}return o}function w1(e,t){let n=-1;const r=[];let i;for(;++n<e.length;){const a=e[n];let o;if(typeof a=="string")o=a;else switch(a){case-5:{o="\r";break}case-4:{o=`
`;break}case-3:{o=`\r
`;break}case-2:{o=t?" ":"	";break}case-1:{if(!t&&i)continue;o=" ";break}default:o=String.fromCharCode(a)}i=a===-2,r.push(o)}return r.join("")}function k1(e){const r={constructs:N_([x1,...(e||{}).extensions||[]]),content:i(M_),defined:[],document:i(F_),flow:i(r1),lazy:{},string:i(o1),text:i(s1)};return r;function i(a){return o;function o(s){return _1(r,a,s)}}}function b1(e){for(;!$m(e););return e}const Sd=/[\0\t\n\r]/g;function S1(){let e=1,t="",n=!0,r;return i;function i(a,o,s){const l=[];let u,c,d,p,f;for(a=t+(typeof a=="string"?a.toString():new TextDecoder(o||void 0).decode(a)),d=0,t="",n&&(a.charCodeAt(0)===65279&&d++,n=void 0);d<a.length;){if(Sd.lastIndex=d,u=Sd.exec(a),p=u&&u.index!==void 0?u.index:a.length,f=a.charCodeAt(p),!u){t=a.slice(d);break}if(f===10&&d===p&&r)l.push(-3),r=void 0;else switch(r&&(l.push(-5),r=void 0),d<p&&(l.push(a.slice(d,p)),e+=p-d),f){case 0:{l.push(65533),e++;break}case 9:{for(c=Math.ceil(e/4)*4,l.push(-2);e++<c;)l.push(-1);break}case 10:{l.push(-4),e=1;break}default:r=!0,e=1}d=p+1}return s&&(r&&l.push(-5),t&&l.push(t),l.push(null)),l}}const C1=/\\([!-/:-@[-`{-~])|&(#(?:\d{1,7}|x[\da-f]{1,6})|[\da-z]{1,31});/gi;function z1(e){return e.replace(C1,E1)}function E1(e,t,n){if(t)return t;if(n.charCodeAt(0)===35){const i=n.charCodeAt(1),a=i===120||i===88;return qm(n.slice(a?2:1),a?16:10)}return Nu(n)||e}const Zm={}.hasOwnProperty;function P1(e,t,n){return typeof t!="string"&&(n=t,t=void 0),N1(n)(b1(k1(n).document().write(S1()(e,t,!0))))}function N1(e){const t={transforms:[],canContainEols:["emphasis","fragment","heading","paragraph","strong"],enter:{autolink:a(Du),autolinkProtocol:O,autolinkEmail:O,atxHeading:a(Nt),blockQuote:a(M),characterEscape:O,characterReference:O,codeFenced:a(H),codeFencedFenceInfo:o,codeFencedFenceMeta:o,codeIndented:a(H,o),codeText:a(Ce,o),codeTextData:O,data:O,codeFlowValue:O,definition:a(xe),definitionDestinationString:o,definitionLabelString:o,definitionTitleString:o,emphasis:a(Fe),hardBreakEscape:a(En),hardBreakTrailing:a(En),htmlFlow:a(Pn,o),htmlFlowData:O,htmlText:a(Pn,o),htmlTextData:O,image:a(Sh),label:o,link:a(Du),listItem:a(Ch),listItemValue:p,listOrdered:a(Fu,d),listUnordered:a(Fu),paragraph:a(zh),reference:v,referenceString:o,resourceDestinationString:o,resourceTitleString:o,setextHeading:a(Nt),strong:a(Eh),thematicBreak:a(Nh)},exit:{atxHeading:l(),atxHeadingSequence:z,autolink:l(),autolinkEmail:T,autolinkProtocol:W,blockQuote:l(),characterEscapeValue:L,characterReferenceMarkerHexadecimal:we,characterReferenceMarkerNumeric:we,characterReferenceValue:se,characterReference:lt,codeFenced:l(S),codeFencedFence:_,codeFencedFenceInfo:f,codeFencedFenceMeta:w,codeFlowValue:L,codeIndented:l(h),codeText:l(j),codeTextData:L,data:L,definition:l(),definitionDestinationString:b,definitionLabelString:m,definitionTitleString:g,emphasis:l(),hardBreakEscape:l(U),hardBreakTrailing:l(U),htmlFlow:l(Y),htmlFlowData:L,htmlText:l(re),htmlTextData:L,image:l(ee),label:V,labelText:I,lineEnding:F,link:l(Q),listItem:l(),listOrdered:l(),listUnordered:l(),paragraph:l(),referenceString:te,resourceDestinationString:y,resourceTitleString:B,resource:$,setextHeading:l(N),setextHeadingLineSequence:E,setextHeadingText:k,strong:l(),thematicBreak:l()}};Jm(t,(e||{}).mdastExtensions||[]);const n={};return r;function r(P){let D={type:"root",children:[]};const K={stack:[D],tokenStack:[],config:t,enter:s,exit:u,buffer:o,resume:c,data:n},J=[];let le=-1;for(;++le<P.length;)if(P[le][1].type==="listOrdered"||P[le][1].type==="listUnordered")if(P[le][0]==="enter")J.push(le);else{const yt=J.pop();le=i(P,yt,le)}for(le=-1;++le<P.length;){const yt=t[P[le][0]];Zm.call(yt,P[le][1].type)&&yt[P[le][1].type].call(Object.assign({sliceSerialize:P[le][2].sliceSerialize},K),P[le][1])}if(K.tokenStack.length>0){const yt=K.tokenStack[K.tokenStack.length-1];(yt[1]||Cd).call(K,void 0,yt[0])}for(D.position={start:Jt(P.length>0?P[0][1].start:{line:1,column:1,offset:0}),end:Jt(P.length>0?P[P.length-2][1].end:{line:1,column:1,offset:0})},le=-1;++le<t.transforms.length;)D=t.transforms[le](D)||D;return D}function i(P,D,K){let J=D-1,le=-1,yt=!1,Nn,Ft,Lr,Ar;for(;++J<=K;){const Ze=P[J];switch(Ze[1].type){case"listUnordered":case"listOrdered":case"blockQuote":{Ze[0]==="enter"?le++:le--,Ar=void 0;break}case"lineEndingBlank":{Ze[0]==="enter"&&(Nn&&!Ar&&!le&&!Lr&&(Lr=J),Ar=void 0);break}case"linePrefix":case"listItemValue":case"listItemMarker":case"listItemPrefix":case"listItemPrefixWhitespace":break;default:Ar=void 0}if(!le&&Ze[0]==="enter"&&Ze[1].type==="listItemPrefix"||le===-1&&Ze[0]==="exit"&&(Ze[1].type==="listUnordered"||Ze[1].type==="listOrdered")){if(Nn){let $n=J;for(Ft=void 0;$n--;){const Wt=P[$n];if(Wt[1].type==="lineEnding"||Wt[1].type==="lineEndingBlank"){if(Wt[0]==="exit")continue;Ft&&(P[Ft][1].type="lineEndingBlank",yt=!0),Wt[1].type="lineEnding",Ft=$n}else if(!(Wt[1].type==="linePrefix"||Wt[1].type==="blockQuotePrefix"||Wt[1].type==="blockQuotePrefixWhitespace"||Wt[1].type==="blockQuoteMarker"||Wt[1].type==="listItemIndent"))break}Lr&&(!Ft||Lr<Ft)&&(Nn._spread=!0),Nn.end=Object.assign({},Ft?P[Ft][1].start:Ze[1].end),P.splice(Ft||J,0,["exit",Nn,Ze[2]]),J++,K++}if(Ze[1].type==="listItemPrefix"){const $n={type:"listItem",_spread:!1,start:Object.assign({},Ze[1].start),end:void 0};Nn=$n,P.splice(J,0,["enter",$n,Ze[2]]),J++,K++,Lr=void 0,Ar=!0}}}return P[D][1]._spread=yt,K}function a(P,D){return K;function K(J){s.call(this,P(J),J),D&&D.call(this,J)}}function o(){this.stack.push({type:"fragment",children:[]})}function s(P,D,K){this.stack[this.stack.length-1].children.push(P),this.stack.push(P),this.tokenStack.push([D,K||void 0]),P.position={start:Jt(D.start),end:void 0}}function l(P){return D;function D(K){P&&P.call(this,K),u.call(this,K)}}function u(P,D){const K=this.stack.pop(),J=this.tokenStack.pop();if(J)J[0].type!==P.type&&(D?D.call(this,P,J[0]):(J[1]||Cd).call(this,P,J[0]));else throw new Error("Cannot close `"+P.type+"` ("+ii({start:P.start,end:P.end})+"): it’s not open");K.position.end=Jt(P.end)}function c(){return E_(this.stack.pop())}function d(){this.data.expectingFirstListItemValue=!0}function p(P){if(this.data.expectingFirstListItemValue){const D=this.stack[this.stack.length-2];D.start=Number.parseInt(this.sliceSerialize(P),10),this.data.expectingFirstListItemValue=void 0}}function f(){const P=this.resume(),D=this.stack[this.stack.length-1];D.lang=P}function w(){const P=this.resume(),D=this.stack[this.stack.length-1];D.meta=P}function _(){this.data.flowCodeInside||(this.buffer(),this.data.flowCodeInside=!0)}function S(){const P=this.resume(),D=this.stack[this.stack.length-1];D.value=P.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g,""),this.data.flowCodeInside=void 0}function h(){const P=this.resume(),D=this.stack[this.stack.length-1];D.value=P.replace(/(\r?\n|\r)$/g,"")}function m(P){const D=this.resume(),K=this.stack[this.stack.length-1];K.label=D,K.identifier=hr(this.sliceSerialize(P)).toLowerCase()}function g(){const P=this.resume(),D=this.stack[this.stack.length-1];D.title=P}function b(){const P=this.resume(),D=this.stack[this.stack.length-1];D.url=P}function z(P){const D=this.stack[this.stack.length-1];if(!D.depth){const K=this.sliceSerialize(P).length;D.depth=K}}function k(){this.data.setextHeadingSlurpLineEnding=!0}function E(P){const D=this.stack[this.stack.length-1];D.depth=this.sliceSerialize(P).codePointAt(0)===61?1:2}function N(){this.data.setextHeadingSlurpLineEnding=void 0}function O(P){const K=this.stack[this.stack.length-1].children;let J=K[K.length-1];(!J||J.type!=="text")&&(J=Ph(),J.position={start:Jt(P.start),end:void 0},K.push(J)),this.stack.push(J)}function L(P){const D=this.stack.pop();D.value+=this.sliceSerialize(P),D.position.end=Jt(P.end)}function F(P){const D=this.stack[this.stack.length-1];if(this.data.atHardBreak){const K=D.children[D.children.length-1];K.position.end=Jt(P.end),this.data.atHardBreak=void 0;return}!this.data.setextHeadingSlurpLineEnding&&t.canContainEols.includes(D.type)&&(O.call(this,P),L.call(this,P))}function U(){this.data.atHardBreak=!0}function Y(){const P=this.resume(),D=this.stack[this.stack.length-1];D.value=P}function re(){const P=this.resume(),D=this.stack[this.stack.length-1];D.value=P}function j(){const P=this.resume(),D=this.stack[this.stack.length-1];D.value=P}function Q(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const D=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=D,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function ee(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const D=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=D,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function I(P){const D=this.sliceSerialize(P),K=this.stack[this.stack.length-2];K.label=z1(D),K.identifier=hr(D).toLowerCase()}function V(){const P=this.stack[this.stack.length-1],D=this.resume(),K=this.stack[this.stack.length-1];if(this.data.inReference=!0,K.type==="link"){const J=P.children;K.children=J}else K.alt=D}function y(){const P=this.resume(),D=this.stack[this.stack.length-1];D.url=P}function B(){const P=this.resume(),D=this.stack[this.stack.length-1];D.title=P}function $(){this.data.inReference=void 0}function v(){this.data.referenceType="collapsed"}function te(P){const D=this.resume(),K=this.stack[this.stack.length-1];K.label=D,K.identifier=hr(this.sliceSerialize(P)).toLowerCase(),this.data.referenceType="full"}function we(P){this.data.characterReferenceType=P.type}function se(P){const D=this.sliceSerialize(P),K=this.data.characterReferenceType;let J;K?(J=qm(D,K==="characterReferenceMarkerNumeric"?10:16),this.data.characterReferenceType=void 0):J=Nu(D);const le=this.stack[this.stack.length-1];le.value+=J}function lt(P){const D=this.stack.pop();D.position.end=Jt(P.end)}function W(P){L.call(this,P);const D=this.stack[this.stack.length-1];D.url=this.sliceSerialize(P)}function T(P){L.call(this,P);const D=this.stack[this.stack.length-1];D.url="mailto:"+this.sliceSerialize(P)}function M(){return{type:"blockquote",children:[]}}function H(){return{type:"code",lang:null,meta:null,value:""}}function Ce(){return{type:"inlineCode",value:""}}function xe(){return{type:"definition",identifier:"",label:null,title:null,url:""}}function Fe(){return{type:"emphasis",children:[]}}function Nt(){return{type:"heading",depth:0,children:[]}}function En(){return{type:"break"}}function Pn(){return{type:"html",value:""}}function Sh(){return{type:"image",title:null,url:"",alt:null}}function Du(){return{type:"link",title:null,url:"",children:[]}}function Fu(P){return{type:"list",ordered:P.type==="listOrdered",start:null,spread:P._spread,children:[]}}function Ch(P){return{type:"listItem",spread:P._spread,checked:null,children:[]}}function zh(){return{type:"paragraph",children:[]}}function Eh(){return{type:"strong",children:[]}}function Ph(){return{type:"text",value:""}}function Nh(){return{type:"thematicBreak"}}}function Jt(e){return{line:e.line,column:e.column,offset:e.offset}}function Jm(e,t){let n=-1;for(;++n<t.length;){const r=t[n];Array.isArray(r)?Jm(e,r):T1(e,r)}}function T1(e,t){let n;for(n in t)if(Zm.call(t,n))switch(n){case"canContainEols":{const r=t[n];r&&e[n].push(...r);break}case"transforms":{const r=t[n];r&&e[n].push(...r);break}case"enter":case"exit":{const r=t[n];r&&Object.assign(e[n],r);break}}}function Cd(e,t){throw e?new Error("Cannot close `"+e.type+"` ("+ii({start:e.start,end:e.end})+"): a different token (`"+t.type+"`, "+ii({start:t.start,end:t.end})+") is open"):new Error("Cannot close document, a token (`"+t.type+"`, "+ii({start:t.start,end:t.end})+") is still open")}function L1(e){const t=this;t.parser=n;function n(r){return P1(r,{...t.data("settings"),...e,extensions:t.data("micromarkExtensions")||[],mdastExtensions:t.data("fromMarkdownExtensions")||[]})}}function A1(e,t){const n={type:"element",tagName:"blockquote",properties:{},children:e.wrap(e.all(t),!0)};return e.patch(t,n),e.applyData(t,n)}function R1(e,t){const n={type:"element",tagName:"br",properties:{},children:[]};return e.patch(t,n),[e.applyData(t,n),{type:"text",value:`
`}]}function I1(e,t){const n=t.value?t.value+`
`:"",r={},i=t.lang?t.lang.split(/\s+/):[];i.length>0&&(r.className=["language-"+i[0]]);let a={type:"element",tagName:"code",properties:r,children:[{type:"text",value:n}]};return t.meta&&(a.data={meta:t.meta}),e.patch(t,a),a=e.applyData(t,a),a={type:"element",tagName:"pre",properties:{},children:[a]},e.patch(t,a),a}function j1(e,t){const n={type:"element",tagName:"del",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function O1(e,t){const n={type:"element",tagName:"em",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function M1(e,t){const n=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",r=String(t.identifier).toUpperCase(),i=Tr(r.toLowerCase()),a=e.footnoteOrder.indexOf(r);let o,s=e.footnoteCounts.get(r);s===void 0?(s=0,e.footnoteOrder.push(r),o=e.footnoteOrder.length):o=a+1,s+=1,e.footnoteCounts.set(r,s);const l={type:"element",tagName:"a",properties:{href:"#"+n+"fn-"+i,id:n+"fnref-"+i+(s>1?"-"+s:""),dataFootnoteRef:!0,ariaDescribedBy:["footnote-label"]},children:[{type:"text",value:String(o)}]};e.patch(t,l);const u={type:"element",tagName:"sup",properties:{},children:[l]};return e.patch(t,u),e.applyData(t,u)}function D1(e,t){const n={type:"element",tagName:"h"+t.depth,properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function F1(e,t){if(e.options.allowDangerousHtml){const n={type:"raw",value:t.value};return e.patch(t,n),e.applyData(t,n)}}function eh(e,t){const n=t.referenceType;let r="]";if(n==="collapsed"?r+="[]":n==="full"&&(r+="["+(t.label||t.identifier)+"]"),t.type==="imageReference")return[{type:"text",value:"!["+t.alt+r}];const i=e.all(t),a=i[0];a&&a.type==="text"?a.value="["+a.value:i.unshift({type:"text",value:"["});const o=i[i.length-1];return o&&o.type==="text"?o.value+=r:i.push({type:"text",value:r}),i}function W1(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return eh(e,t);const i={src:Tr(r.url||""),alt:t.alt};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"img",properties:i,children:[]};return e.patch(t,a),e.applyData(t,a)}function B1(e,t){const n={src:Tr(t.url)};t.alt!==null&&t.alt!==void 0&&(n.alt=t.alt),t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"img",properties:n,children:[]};return e.patch(t,r),e.applyData(t,r)}function q1(e,t){const n={type:"text",value:t.value.replace(/\r?\n|\r/g," ")};e.patch(t,n);const r={type:"element",tagName:"code",properties:{},children:[n]};return e.patch(t,r),e.applyData(t,r)}function U1(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return eh(e,t);const i={href:Tr(r.url||"")};r.title!==null&&r.title!==void 0&&(i.title=r.title);const a={type:"element",tagName:"a",properties:i,children:e.all(t)};return e.patch(t,a),e.applyData(t,a)}function V1(e,t){const n={href:Tr(t.url)};t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"a",properties:n,children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function H1(e,t,n){const r=e.all(t),i=n?$1(n):th(t),a={},o=[];if(typeof t.checked=="boolean"){const c=r[0];let d;c&&c.type==="element"&&c.tagName==="p"?d=c:(d={type:"element",tagName:"p",properties:{},children:[]},r.unshift(d)),d.children.length>0&&d.children.unshift({type:"text",value:" "}),d.children.unshift({type:"element",tagName:"input",properties:{type:"checkbox",checked:t.checked,disabled:!0},children:[]}),a.className=["task-list-item"]}let s=-1;for(;++s<r.length;){const c=r[s];(i||s!==0||c.type!=="element"||c.tagName!=="p")&&o.push({type:"text",value:`
`}),c.type==="element"&&c.tagName==="p"&&!i?o.push(...c.children):o.push(c)}const l=r[r.length-1];l&&(i||l.type!=="element"||l.tagName!=="p")&&o.push({type:"text",value:`
`});const u={type:"element",tagName:"li",properties:a,children:o};return e.patch(t,u),e.applyData(t,u)}function $1(e){let t=!1;if(e.type==="list"){t=e.spread||!1;const n=e.children;let r=-1;for(;!t&&++r<n.length;)t=th(n[r])}return t}function th(e){const t=e.spread;return t??e.children.length>1}function Q1(e,t){const n={},r=e.all(t);let i=-1;for(typeof t.start=="number"&&t.start!==1&&(n.start=t.start);++i<r.length;){const o=r[i];if(o.type==="element"&&o.tagName==="li"&&o.properties&&Array.isArray(o.properties.className)&&o.properties.className.includes("task-list-item")){n.className=["contains-task-list"];break}}const a={type:"element",tagName:t.ordered?"ol":"ul",properties:n,children:e.wrap(r,!0)};return e.patch(t,a),e.applyData(t,a)}function X1(e,t){const n={type:"element",tagName:"p",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function K1(e,t){const n={type:"root",children:e.wrap(e.all(t))};return e.patch(t,n),e.applyData(t,n)}function Y1(e,t){const n={type:"element",tagName:"strong",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function G1(e,t){const n=e.all(t),r=n.shift(),i=[];if(r){const o={type:"element",tagName:"thead",properties:{},children:e.wrap([r],!0)};e.patch(t.children[0],o),i.push(o)}if(n.length>0){const o={type:"element",tagName:"tbody",properties:{},children:e.wrap(n,!0)},s=Cu(t.children[1]),l=jm(t.children[t.children.length-1]);s&&l&&(o.position={start:s,end:l}),i.push(o)}const a={type:"element",tagName:"table",properties:{},children:e.wrap(i,!0)};return e.patch(t,a),e.applyData(t,a)}function Z1(e,t,n){const r=n?n.children:void 0,a=(r?r.indexOf(t):1)===0?"th":"td",o=n&&n.type==="table"?n.align:void 0,s=o?o.length:t.children.length;let l=-1;const u=[];for(;++l<s;){const d=t.children[l],p={},f=o?o[l]:void 0;f&&(p.align=f);let w={type:"element",tagName:a,properties:p,children:[]};d&&(w.children=e.all(d),e.patch(d,w),w=e.applyData(d,w)),u.push(w)}const c={type:"element",tagName:"tr",properties:{},children:e.wrap(u,!0)};return e.patch(t,c),e.applyData(t,c)}function J1(e,t){const n={type:"element",tagName:"td",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}const zd=9,Ed=32;function ew(e){const t=String(e),n=/\r?\n|\r/g;let r=n.exec(t),i=0;const a=[];for(;r;)a.push(Pd(t.slice(i,r.index),i>0,!0),r[0]),i=r.index+r[0].length,r=n.exec(t);return a.push(Pd(t.slice(i),i>0,!1)),a.join("")}function Pd(e,t,n){let r=0,i=e.length;if(t){let a=e.codePointAt(r);for(;a===zd||a===Ed;)r++,a=e.codePointAt(r)}if(n){let a=e.codePointAt(i-1);for(;a===zd||a===Ed;)i--,a=e.codePointAt(i-1)}return i>r?e.slice(r,i):""}function tw(e,t){const n={type:"text",value:ew(String(t.value))};return e.patch(t,n),e.applyData(t,n)}function nw(e,t){const n={type:"element",tagName:"hr",properties:{},children:[]};return e.patch(t,n),e.applyData(t,n)}const rw={blockquote:A1,break:R1,code:I1,delete:j1,emphasis:O1,footnoteReference:M1,heading:D1,html:F1,imageReference:W1,image:B1,inlineCode:q1,linkReference:U1,link:V1,listItem:H1,list:Q1,paragraph:X1,root:K1,strong:Y1,table:G1,tableCell:J1,tableRow:Z1,text:tw,thematicBreak:nw,toml:ia,yaml:ia,definition:ia,footnoteDefinition:ia};function ia(){}const nh=-1,zo=0,oi=1,Ja=2,Au=3,Ru=4,Iu=5,ju=6,rh=7,ih=8,Nd=typeof self=="object"?self:globalThis,iw=(e,t)=>{const n=(i,a)=>(e.set(a,i),i),r=i=>{if(e.has(i))return e.get(i);const[a,o]=t[i];switch(a){case zo:case nh:return n(o,i);case oi:{const s=n([],i);for(const l of o)s.push(r(l));return s}case Ja:{const s=n({},i);for(const[l,u]of o)s[r(l)]=r(u);return s}case Au:return n(new Date(o),i);case Ru:{const{source:s,flags:l}=o;return n(new RegExp(s,l),i)}case Iu:{const s=n(new Map,i);for(const[l,u]of o)s.set(r(l),r(u));return s}case ju:{const s=n(new Set,i);for(const l of o)s.add(r(l));return s}case rh:{const{name:s,message:l}=o;return n(new Nd[s](l),i)}case ih:return n(BigInt(o),i);case"BigInt":return n(Object(BigInt(o)),i);case"ArrayBuffer":return n(new Uint8Array(o).buffer,o);case"DataView":{const{buffer:s}=new Uint8Array(o);return n(new DataView(s),o)}}return n(new Nd[a](o),i)};return r},Td=e=>iw(new Map,e)(0),Xn="",{toString:aw}={},{keys:ow}=Object,qr=e=>{const t=typeof e;if(t!=="object"||!e)return[zo,t];const n=aw.call(e).slice(8,-1);switch(n){case"Array":return[oi,Xn];case"Object":return[Ja,Xn];case"Date":return[Au,Xn];case"RegExp":return[Ru,Xn];case"Map":return[Iu,Xn];case"Set":return[ju,Xn];case"DataView":return[oi,n]}return n.includes("Array")?[oi,n]:n.includes("Error")?[rh,n]:[Ja,n]},aa=([e,t])=>e===zo&&(t==="function"||t==="symbol"),sw=(e,t,n,r)=>{const i=(o,s)=>{const l=r.push(o)-1;return n.set(s,l),l},a=o=>{if(n.has(o))return n.get(o);let[s,l]=qr(o);switch(s){case zo:{let c=o;switch(l){case"bigint":s=ih,c=o.toString();break;case"function":case"symbol":if(e)throw new TypeError("unable to serialize "+l);c=null;break;case"undefined":return i([nh],o)}return i([s,c],o)}case oi:{if(l){let p=o;return l==="DataView"?p=new Uint8Array(o.buffer):l==="ArrayBuffer"&&(p=new Uint8Array(o)),i([l,[...p]],o)}const c=[],d=i([s,c],o);for(const p of o)c.push(a(p));return d}case Ja:{if(l)switch(l){case"BigInt":return i([l,o.toString()],o);case"Boolean":case"Number":case"String":return i([l,o.valueOf()],o)}if(t&&"toJSON"in o)return a(o.toJSON());const c=[],d=i([s,c],o);for(const p of ow(o))(e||!aa(qr(o[p])))&&c.push([a(p),a(o[p])]);return d}case Au:return i([s,o.toISOString()],o);case Ru:{const{source:c,flags:d}=o;return i([s,{source:c,flags:d}],o)}case Iu:{const c=[],d=i([s,c],o);for(const[p,f]of o)(e||!(aa(qr(p))||aa(qr(f))))&&c.push([a(p),a(f)]);return d}case ju:{const c=[],d=i([s,c],o);for(const p of o)(e||!aa(qr(p)))&&c.push(a(p));return d}}const{message:u}=o;return i([s,{name:l,message:u}],o)};return a},Ld=(e,{json:t,lossy:n}={})=>{const r=[];return sw(!(t||n),!!t,new Map,r)(e),r},eo=typeof structuredClone=="function"?(e,t)=>t&&("json"in t||"lossy"in t)?Td(Ld(e,t)):structuredClone(e):(e,t)=>Td(Ld(e,t));function lw(e,t){const n=[{type:"text",value:"↩"}];return t>1&&n.push({type:"element",tagName:"sup",properties:{},children:[{type:"text",value:String(t)}]}),n}function uw(e,t){return"Back to reference "+(e+1)+(t>1?"-"+t:"")}function cw(e){const t=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",n=e.options.footnoteBackContent||lw,r=e.options.footnoteBackLabel||uw,i=e.options.footnoteLabel||"Footnotes",a=e.options.footnoteLabelTagName||"h2",o=e.options.footnoteLabelProperties||{className:["sr-only"]},s=[];let l=-1;for(;++l<e.footnoteOrder.length;){const u=e.footnoteById.get(e.footnoteOrder[l]);if(!u)continue;const c=e.all(u),d=String(u.identifier).toUpperCase(),p=Tr(d.toLowerCase());let f=0;const w=[],_=e.footnoteCounts.get(d);for(;_!==void 0&&++f<=_;){w.length>0&&w.push({type:"text",value:" "});let m=typeof n=="string"?n:n(l,f);typeof m=="string"&&(m={type:"text",value:m}),w.push({type:"element",tagName:"a",properties:{href:"#"+t+"fnref-"+p+(f>1?"-"+f:""),dataFootnoteBackref:"",ariaLabel:typeof r=="string"?r:r(l,f),className:["data-footnote-backref"]},children:Array.isArray(m)?m:[m]})}const S=c[c.length-1];if(S&&S.type==="element"&&S.tagName==="p"){const m=S.children[S.children.length-1];m&&m.type==="text"?m.value+=" ":S.children.push({type:"text",value:" "}),S.children.push(...w)}else c.push(...w);const h={type:"element",tagName:"li",properties:{id:t+"fn-"+p},children:e.wrap(c,!0)};e.patch(u,h),s.push(h)}if(s.length!==0)return{type:"element",tagName:"section",properties:{dataFootnotes:!0,className:["footnotes"]},children:[{type:"element",tagName:a,properties:{...eo(o),id:"footnote-label"},children:[{type:"text",value:i}]},{type:"text",value:`
`},{type:"element",tagName:"ol",properties:{},children:e.wrap(s,!0)},{type:"text",value:`
`}]}}const ah=function(e){if(e==null)return mw;if(typeof e=="function")return Eo(e);if(typeof e=="object")return Array.isArray(e)?dw(e):pw(e);if(typeof e=="string")return fw(e);throw new Error("Expected function, string, or object as test")};function dw(e){const t=[];let n=-1;for(;++n<e.length;)t[n]=ah(e[n]);return Eo(r);function r(...i){let a=-1;for(;++a<t.length;)if(t[a].apply(this,i))return!0;return!1}}function pw(e){const t=e;return Eo(n);function n(r){const i=r;let a;for(a in e)if(i[a]!==t[a])return!1;return!0}}function fw(e){return Eo(t);function t(n){return n&&n.type===e}}function Eo(e){return t;function t(n,r,i){return!!(hw(n)&&e.call(this,n,typeof r=="number"?r:void 0,i||void 0))}}function mw(){return!0}function hw(e){return e!==null&&typeof e=="object"&&"type"in e}const oh=[],gw=!0,Ad=!1,yw="skip";function xw(e,t,n,r){let i;typeof t=="function"&&typeof n!="function"?(r=n,n=t):i=t;const a=ah(i),o=r?-1:1;s(e,void 0,[])();function s(l,u,c){const d=l&&typeof l=="object"?l:{};if(typeof d.type=="string"){const f=typeof d.tagName=="string"?d.tagName:typeof d.name=="string"?d.name:void 0;Object.defineProperty(p,"name",{value:"node ("+(l.type+(f?"<"+f+">":""))+")"})}return p;function p(){let f=oh,w,_,S;if((!t||a(l,u,c[c.length-1]||void 0))&&(f=_w(n(l,c)),f[0]===Ad))return f;if("children"in l&&l.children){const h=l;if(h.children&&f[0]!==yw)for(_=(r?h.children.length:-1)+o,S=c.concat(h);_>-1&&_<h.children.length;){const m=h.children[_];if(w=s(m,_,S)(),w[0]===Ad)return w;_=typeof w[1]=="number"?w[1]:_+o}}return f}}}function _w(e){return Array.isArray(e)?e:typeof e=="number"?[gw,e]:e==null?oh:[e]}function sh(e,t,n,r){let i,a,o;typeof t=="function"&&typeof n!="function"?(a=void 0,o=t,i=n):(a=t,o=n,i=r),xw(e,a,s,i);function s(l,u){const c=u[u.length-1],d=c?c.children.indexOf(l):void 0;return o(l,d,c)}}const Sl={}.hasOwnProperty,vw={};function ww(e,t){const n=t||vw,r=new Map,i=new Map,a=new Map,o={...rw,...n.handlers},s={all:u,applyData:bw,definitionById:r,footnoteById:i,footnoteCounts:a,footnoteOrder:[],handlers:o,one:l,options:n,patch:kw,wrap:Cw};return sh(e,function(c){if(c.type==="definition"||c.type==="footnoteDefinition"){const d=c.type==="definition"?r:i,p=String(c.identifier).toUpperCase();d.has(p)||d.set(p,c)}}),s;function l(c,d){const p=c.type,f=s.handlers[p];if(Sl.call(s.handlers,p)&&f)return f(s,c,d);if(s.options.passThrough&&s.options.passThrough.includes(p)){if("children"in c){const{children:_,...S}=c,h=eo(S);return h.children=s.all(c),h}return eo(c)}return(s.options.unknownHandler||Sw)(s,c,d)}function u(c){const d=[];if("children"in c){const p=c.children;let f=-1;for(;++f<p.length;){const w=s.one(p[f],c);if(w){if(f&&p[f-1].type==="break"&&(!Array.isArray(w)&&w.type==="text"&&(w.value=Rd(w.value)),!Array.isArray(w)&&w.type==="element")){const _=w.children[0];_&&_.type==="text"&&(_.value=Rd(_.value))}Array.isArray(w)?d.push(...w):d.push(w)}}}return d}}function kw(e,t){e.position&&(t.position=a_(e))}function bw(e,t){let n=t;if(e&&e.data){const r=e.data.hName,i=e.data.hChildren,a=e.data.hProperties;if(typeof r=="string")if(n.type==="element")n.tagName=r;else{const o="children"in n?n.children:[n];n={type:"element",tagName:r,properties:{},children:o}}n.type==="element"&&a&&Object.assign(n.properties,eo(a)),"children"in n&&n.children&&i!==null&&i!==void 0&&(n.children=i)}return n}function Sw(e,t){const n=t.data||{},r="value"in t&&!(Sl.call(n,"hProperties")||Sl.call(n,"hChildren"))?{type:"text",value:t.value}:{type:"element",tagName:"div",properties:{},children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function Cw(e,t){const n=[];let r=-1;for(t&&n.push({type:"text",value:`
`});++r<e.length;)r&&n.push({type:"text",value:`
`}),n.push(e[r]);return t&&e.length>0&&n.push({type:"text",value:`
`}),n}function Rd(e){let t=0,n=e.charCodeAt(t);for(;n===9||n===32;)t++,n=e.charCodeAt(t);return e.slice(t)}function Id(e,t){const n=ww(e,t),r=n.one(e,void 0),i=cw(n),a=Array.isArray(r)?{type:"root",children:r}:r||{type:"root",children:[]};return i&&a.children.push({type:"text",value:`
`},i),a}function zw(e,t){return e&&"run"in e?async function(n,r){const i=Id(n,{file:r,...t});await e.run(i,r)}:function(n,r){return Id(n,{file:r,...e||t})}}function jd(e){if(e)throw e}var Ca=Object.prototype.hasOwnProperty,lh=Object.prototype.toString,Od=Object.defineProperty,Md=Object.getOwnPropertyDescriptor,Dd=function(t){return typeof Array.isArray=="function"?Array.isArray(t):lh.call(t)==="[object Array]"},Fd=function(t){if(!t||lh.call(t)!=="[object Object]")return!1;var n=Ca.call(t,"constructor"),r=t.constructor&&t.constructor.prototype&&Ca.call(t.constructor.prototype,"isPrototypeOf");if(t.constructor&&!n&&!r)return!1;var i;for(i in t);return typeof i>"u"||Ca.call(t,i)},Wd=function(t,n){Od&&n.name==="__proto__"?Od(t,n.name,{enumerable:!0,configurable:!0,value:n.newValue,writable:!0}):t[n.name]=n.newValue},Bd=function(t,n){if(n==="__proto__")if(Ca.call(t,n)){if(Md)return Md(t,n).value}else return;return t[n]},Ew=function e(){var t,n,r,i,a,o,s=arguments[0],l=1,u=arguments.length,c=!1;for(typeof s=="boolean"&&(c=s,s=arguments[1]||{},l=2),(s==null||typeof s!="object"&&typeof s!="function")&&(s={});l<u;++l)if(t=arguments[l],t!=null)for(n in t)r=Bd(s,n),i=Bd(t,n),s!==i&&(c&&i&&(Fd(i)||(a=Dd(i)))?(a?(a=!1,o=r&&Dd(r)?r:[]):o=r&&Fd(r)?r:{},Wd(s,{name:n,newValue:e(c,o,i)})):typeof i<"u"&&Wd(s,{name:n,newValue:i}));return s};const ss=ro(Ew);function Cl(e){if(typeof e!="object"||e===null)return!1;const t=Object.getPrototypeOf(e);return(t===null||t===Object.prototype||Object.getPrototypeOf(t)===null)&&!(Symbol.toStringTag in e)&&!(Symbol.iterator in e)}function Pw(){const e=[],t={run:n,use:r};return t;function n(...i){let a=-1;const o=i.pop();if(typeof o!="function")throw new TypeError("Expected function as last argument, not "+o);s(null,...i);function s(l,...u){const c=e[++a];let d=-1;if(l){o(l);return}for(;++d<i.length;)(u[d]===null||u[d]===void 0)&&(u[d]=i[d]);i=u,c?Nw(c,s)(...u):o(null,...u)}}function r(i){if(typeof i!="function")throw new TypeError("Expected `middelware` to be a function, not "+i);return e.push(i),t}}function Nw(e,t){let n;return r;function r(...o){const s=e.length>o.length;let l;s&&o.push(i);try{l=e.apply(this,o)}catch(u){const c=u;if(s&&n)throw c;return i(c)}s||(l&&l.then&&typeof l.then=="function"?l.then(a,i):l instanceof Error?i(l):a(l))}function i(o,...s){n||(n=!0,t(o,...s))}function a(o){i(null,o)}}const At={basename:Tw,dirname:Lw,extname:Aw,join:Rw,sep:"/"};function Tw(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');Fi(e);let n=0,r=-1,i=e.length,a;if(t===void 0||t.length===0||t.length>e.length){for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else r<0&&(a=!0,r=i+1);return r<0?"":e.slice(n,r)}if(t===e)return"";let o=-1,s=t.length-1;for(;i--;)if(e.codePointAt(i)===47){if(a){n=i+1;break}}else o<0&&(a=!0,o=i+1),s>-1&&(e.codePointAt(i)===t.codePointAt(s--)?s<0&&(r=i):(s=-1,r=o));return n===r?r=o:r<0&&(r=e.length),e.slice(n,r)}function Lw(e){if(Fi(e),e.length===0)return".";let t=-1,n=e.length,r;for(;--n;)if(e.codePointAt(n)===47){if(r){t=n;break}}else r||(r=!0);return t<0?e.codePointAt(0)===47?"/":".":t===1&&e.codePointAt(0)===47?"//":e.slice(0,t)}function Aw(e){Fi(e);let t=e.length,n=-1,r=0,i=-1,a=0,o;for(;t--;){const s=e.codePointAt(t);if(s===47){if(o){r=t+1;break}continue}n<0&&(o=!0,n=t+1),s===46?i<0?i=t:a!==1&&(a=1):i>-1&&(a=-1)}return i<0||n<0||a===0||a===1&&i===n-1&&i===r+1?"":e.slice(i,n)}function Rw(...e){let t=-1,n;for(;++t<e.length;)Fi(e[t]),e[t]&&(n=n===void 0?e[t]:n+"/"+e[t]);return n===void 0?".":Iw(n)}function Iw(e){Fi(e);const t=e.codePointAt(0)===47;let n=jw(e,!t);return n.length===0&&!t&&(n="."),n.length>0&&e.codePointAt(e.length-1)===47&&(n+="/"),t?"/"+n:n}function jw(e,t){let n="",r=0,i=-1,a=0,o=-1,s,l;for(;++o<=e.length;){if(o<e.length)s=e.codePointAt(o);else{if(s===47)break;s=47}if(s===47){if(!(i===o-1||a===1))if(i!==o-1&&a===2){if(n.length<2||r!==2||n.codePointAt(n.length-1)!==46||n.codePointAt(n.length-2)!==46){if(n.length>2){if(l=n.lastIndexOf("/"),l!==n.length-1){l<0?(n="",r=0):(n=n.slice(0,l),r=n.length-1-n.lastIndexOf("/")),i=o,a=0;continue}}else if(n.length>0){n="",r=0,i=o,a=0;continue}}t&&(n=n.length>0?n+"/..":"..",r=2)}else n.length>0?n+="/"+e.slice(i+1,o):n=e.slice(i+1,o),r=o-i-1;i=o,a=0}else s===46&&a>-1?a++:a=-1}return n}function Fi(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}const Ow={cwd:Mw};function Mw(){return"/"}function zl(e){return!!(e!==null&&typeof e=="object"&&"href"in e&&e.href&&"protocol"in e&&e.protocol&&e.auth===void 0)}function Dw(e){if(typeof e=="string")e=new URL(e);else if(!zl(e)){const t=new TypeError('The "path" argument must be of type string or an instance of URL. Received `'+e+"`");throw t.code="ERR_INVALID_ARG_TYPE",t}if(e.protocol!=="file:"){const t=new TypeError("The URL must be of scheme file");throw t.code="ERR_INVALID_URL_SCHEME",t}return Fw(e)}function Fw(e){if(e.hostname!==""){const r=new TypeError('File URL host must be "localhost" or empty on darwin');throw r.code="ERR_INVALID_FILE_URL_HOST",r}const t=e.pathname;let n=-1;for(;++n<t.length;)if(t.codePointAt(n)===37&&t.codePointAt(n+1)===50){const r=t.codePointAt(n+2);if(r===70||r===102){const i=new TypeError("File URL path must not include encoded / characters");throw i.code="ERR_INVALID_FILE_URL_PATH",i}}return decodeURIComponent(t)}const ls=["history","path","basename","stem","extname","dirname"];class uh{constructor(t){let n;t?zl(t)?n={path:t}:typeof t=="string"||Ww(t)?n={value:t}:n=t:n={},this.cwd="cwd"in n?"":Ow.cwd(),this.data={},this.history=[],this.messages=[],this.value,this.map,this.result,this.stored;let r=-1;for(;++r<ls.length;){const a=ls[r];a in n&&n[a]!==void 0&&n[a]!==null&&(this[a]=a==="history"?[...n[a]]:n[a])}let i;for(i in n)ls.includes(i)||(this[i]=n[i])}get basename(){return typeof this.path=="string"?At.basename(this.path):void 0}set basename(t){cs(t,"basename"),us(t,"basename"),this.path=At.join(this.dirname||"",t)}get dirname(){return typeof this.path=="string"?At.dirname(this.path):void 0}set dirname(t){qd(this.basename,"dirname"),this.path=At.join(t||"",this.basename)}get extname(){return typeof this.path=="string"?At.extname(this.path):void 0}set extname(t){if(us(t,"extname"),qd(this.dirname,"extname"),t){if(t.codePointAt(0)!==46)throw new Error("`extname` must start with `.`");if(t.includes(".",1))throw new Error("`extname` cannot contain multiple dots")}this.path=At.join(this.dirname,this.stem+(t||""))}get path(){return this.history[this.history.length-1]}set path(t){zl(t)&&(t=Dw(t)),cs(t,"path"),this.path!==t&&this.history.push(t)}get stem(){return typeof this.path=="string"?At.basename(this.path,this.extname):void 0}set stem(t){cs(t,"stem"),us(t,"stem"),this.path=At.join(this.dirname||"",t+(this.extname||""))}fail(t,n,r){const i=this.message(t,n,r);throw i.fatal=!0,i}info(t,n,r){const i=this.message(t,n,r);return i.fatal=void 0,i}message(t,n,r){const i=new De(t,n,r);return this.path&&(i.name=this.path+":"+i.name,i.file=this.path),i.fatal=!1,this.messages.push(i),i}toString(t){return this.value===void 0?"":typeof this.value=="string"?this.value:new TextDecoder(t||void 0).decode(this.value)}}function us(e,t){if(e&&e.includes(At.sep))throw new Error("`"+t+"` cannot be a path: did not expect `"+At.sep+"`")}function cs(e,t){if(!e)throw new Error("`"+t+"` cannot be empty")}function qd(e,t){if(!e)throw new Error("Setting `"+t+"` requires `path` to be set too")}function Ww(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const Bw=function(e){const r=this.constructor.prototype,i=r[e],a=function(){return i.apply(a,arguments)};return Object.setPrototypeOf(a,r),a},qw={}.hasOwnProperty;class Ou extends Bw{constructor(){super("copy"),this.Compiler=void 0,this.Parser=void 0,this.attachers=[],this.compiler=void 0,this.freezeIndex=-1,this.frozen=void 0,this.namespace={},this.parser=void 0,this.transformers=Pw()}copy(){const t=new Ou;let n=-1;for(;++n<this.attachers.length;){const r=this.attachers[n];t.use(...r)}return t.data(ss(!0,{},this.namespace)),t}data(t,n){return typeof t=="string"?arguments.length===2?(fs("data",this.frozen),this.namespace[t]=n,this):qw.call(this.namespace,t)&&this.namespace[t]||void 0:t?(fs("data",this.frozen),this.namespace=t,this):this.namespace}freeze(){if(this.frozen)return this;const t=this;for(;++this.freezeIndex<this.attachers.length;){const[n,...r]=this.attachers[this.freezeIndex];if(r[0]===!1)continue;r[0]===!0&&(r[0]=void 0);const i=n.call(t,...r);typeof i=="function"&&this.transformers.use(i)}return this.frozen=!0,this.freezeIndex=Number.POSITIVE_INFINITY,this}parse(t){this.freeze();const n=oa(t),r=this.parser||this.Parser;return ds("parse",r),r(String(n),n)}process(t,n){const r=this;return this.freeze(),ds("process",this.parser||this.Parser),ps("process",this.compiler||this.Compiler),n?i(void 0,n):new Promise(i);function i(a,o){const s=oa(t),l=r.parse(s);r.run(l,s,function(c,d,p){if(c||!d||!p)return u(c);const f=d,w=r.stringify(f,p);Hw(w)?p.value=w:p.result=w,u(c,p)});function u(c,d){c||!d?o(c):a?a(d):n(void 0,d)}}}processSync(t){let n=!1,r;return this.freeze(),ds("processSync",this.parser||this.Parser),ps("processSync",this.compiler||this.Compiler),this.process(t,i),Vd("processSync","process",n),r;function i(a,o){n=!0,jd(a),r=o}}run(t,n,r){Ud(t),this.freeze();const i=this.transformers;return!r&&typeof n=="function"&&(r=n,n=void 0),r?a(void 0,r):new Promise(a);function a(o,s){const l=oa(n);i.run(t,l,u);function u(c,d,p){const f=d||t;c?s(c):o?o(f):r(void 0,f,p)}}}runSync(t,n){let r=!1,i;return this.run(t,n,a),Vd("runSync","run",r),i;function a(o,s){jd(o),i=s,r=!0}}stringify(t,n){this.freeze();const r=oa(n),i=this.compiler||this.Compiler;return ps("stringify",i),Ud(t),i(t,r)}use(t,...n){const r=this.attachers,i=this.namespace;if(fs("use",this.frozen),t!=null)if(typeof t=="function")l(t,n);else if(typeof t=="object")Array.isArray(t)?s(t):o(t);else throw new TypeError("Expected usable value, not `"+t+"`");return this;function a(u){if(typeof u=="function")l(u,[]);else if(typeof u=="object")if(Array.isArray(u)){const[c,...d]=u;l(c,d)}else o(u);else throw new TypeError("Expected usable value, not `"+u+"`")}function o(u){if(!("plugins"in u)&&!("settings"in u))throw new Error("Expected usable value but received an empty preset, which is probably a mistake: presets typically come with `plugins` and sometimes with `settings`, but this has neither");s(u.plugins),u.settings&&(i.settings=ss(!0,i.settings,u.settings))}function s(u){let c=-1;if(u!=null)if(Array.isArray(u))for(;++c<u.length;){const d=u[c];a(d)}else throw new TypeError("Expected a list of plugins, not `"+u+"`")}function l(u,c){let d=-1,p=-1;for(;++d<r.length;)if(r[d][0]===u){p=d;break}if(p===-1)r.push([u,...c]);else if(c.length>0){let[f,...w]=c;const _=r[p][1];Cl(_)&&Cl(f)&&(f=ss(!0,_,f)),r[p]=[u,f,...w]}}}}const Uw=new Ou().freeze();function ds(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `parser`")}function ps(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `compiler`")}function fs(e,t){if(t)throw new Error("Cannot call `"+e+"` on a frozen processor.\nCreate a new processor first, by calling it: use `processor()` instead of `processor`.")}function Ud(e){if(!Cl(e)||typeof e.type!="string")throw new TypeError("Expected node, got `"+e+"`")}function Vd(e,t,n){if(!n)throw new Error("`"+e+"` finished async. Use `"+t+"` instead")}function oa(e){return Vw(e)?e:new uh(e)}function Vw(e){return!!(e&&typeof e=="object"&&"message"in e&&"messages"in e)}function Hw(e){return typeof e=="string"||$w(e)}function $w(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const Qw="https://github.com/remarkjs/react-markdown/blob/main/changelog.md",Hd=[],$d={allowDangerousHtml:!0},Xw=/^(https?|ircs?|mailto|xmpp)$/i,Kw=[{from:"astPlugins",id:"remove-buggy-html-in-markdown-parser"},{from:"allowDangerousHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"allowNode",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowElement"},{from:"allowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowedElements"},{from:"disallowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"disallowedElements"},{from:"escapeHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"includeElementIndex",id:"#remove-includeelementindex"},{from:"includeNodeIndex",id:"change-includenodeindex-to-includeelementindex"},{from:"linkTarget",id:"remove-linktarget"},{from:"plugins",id:"change-plugins-to-remarkplugins",to:"remarkPlugins"},{from:"rawSourcePos",id:"#remove-rawsourcepos"},{from:"renderers",id:"change-renderers-to-components",to:"components"},{from:"source",id:"change-source-to-children",to:"children"},{from:"sourcePos",id:"#remove-sourcepos"},{from:"transformImageUri",id:"#add-urltransform",to:"urlTransform"},{from:"transformLinkUri",id:"#add-urltransform",to:"urlTransform"}];function ch(e){const t=Yw(e),n=Gw(e);return Zw(t.runSync(t.parse(n),n),e)}function Yw(e){const t=e.rehypePlugins||Hd,n=e.remarkPlugins||Hd,r=e.remarkRehypeOptions?{...e.remarkRehypeOptions,...$d}:$d;return Uw().use(L1).use(n).use(zw,r).use(t)}function Gw(e){const t=e.children||"",n=new uh;return typeof t=="string"&&(n.value=t),n}function Zw(e,t){const n=t.allowedElements,r=t.allowElement,i=t.components,a=t.disallowedElements,o=t.skipHtml,s=t.unwrapDisallowed,l=t.urlTransform||Jw;for(const c of Kw)Object.hasOwn(t,c.from)&&(""+c.from+(c.to?"use `"+c.to+"` instead":"remove it")+Qw+c.id,void 0);return t.className&&(e={type:"element",tagName:"div",properties:{className:t.className},children:e.type==="root"?e.children:[e]}),sh(e,u),c_(e,{Fragment:x.Fragment,components:i,ignoreInvalidStyle:!0,jsx:x.jsx,jsxs:x.jsxs,passKeys:!0,passNode:!0});function u(c,d,p){if(c.type==="raw"&&p&&typeof d=="number")return o?p.children.splice(d,1):p.children[d]={type:"text",value:c.value},d;if(c.type==="element"){let f;for(f in is)if(Object.hasOwn(is,f)&&Object.hasOwn(c.properties,f)){const w=c.properties[f],_=is[f];(_===null||_.includes(c.tagName))&&(c.properties[f]=l(String(w||""),f,c))}}if(c.type==="element"){let f=n?!n.includes(c.tagName):a?a.includes(c.tagName):!1;if(!f&&r&&typeof d=="number"&&(f=!r(c,d,p)),f&&p&&typeof d=="number")return s&&c.children?p.children.splice(d,1,...c.children):p.children.splice(d,1),d}}}function Jw(e){const t=e.indexOf(":"),n=e.indexOf("?"),r=e.indexOf("#"),i=e.indexOf("/");return t===-1||i!==-1&&t>i||n!==-1&&t>n||r!==-1&&t>r||Xw.test(e.slice(0,t))?e:""}const ek=[{id:"numpy-array-sum",title:"Array Sum",section:"python-basics",difficulty:"easy",description:`
## Array Sum

Given a NumPy array of numbers, implement a function that returns the sum of all elements.

### Function Signature
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
\`\`\`

### Constraints
- Array length: 1 <= n <= 1000
- Element values: -10^6 <= arr[i] <= 10^6
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"15",explanation:"1 + 2 + 3 + 4 + 5 = 15"},{input:"np.array([-1, 0, 1])",output:"0",explanation:"-1 + 0 + 1 = 0"}],starterCode:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.

    Args:
        arr: A NumPy array of numbers

    Returns:
        The sum of all elements
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic positive numbers",input:"[1, 2, 3, 4, 5]",expected:"15",hidden:!1},{id:"2",description:"With negatives",input:"[-1, 0, 1]",expected:"0",hidden:!1},{id:"3",description:"Single element",input:"[42]",expected:"42",hidden:!1},{id:"4",description:"Larger array",input:"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",expected:"55",hidden:!0}],hints:["NumPy arrays have built-in methods for common operations.","Try using np.sum() or the .sum() method on the array."],solution:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.
    """
    return np.sum(arr)
`},{id:"numpy-matrix-multiply",title:"Matrix Multiplication",section:"python-basics",difficulty:"medium",description:`
## Matrix Multiplication

Implement matrix multiplication between two 2D NumPy arrays.

### Function Signature
\`\`\`python
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- A has shape (m, n)
- B has shape (n, p)
- Result has shape (m, p)
- 1 <= m, n, p <= 100

### Note
You should use NumPy's built-in functions, not manual loops.
    `,examples:[{input:"A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]",output:"[[19, 22], [43, 50]]",explanation:"Standard matrix multiplication"}],starterCode:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Result matrix of shape (m, p)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2x2 matrices",input:"([[1, 2], [3, 4]], [[5, 6], [7, 8]])",expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Identity multiplication",input:"([[1, 0], [0, 1]], [[5, 6], [7, 8]])",expected:"[[5, 6], [7, 8]]",hidden:!1},{id:"3",description:"Different dimensions",input:"([[1, 2, 3]], [[1], [2], [3]])",expected:"[[14]]",hidden:!0}],hints:["NumPy provides several ways to do matrix multiplication.","Try np.dot(), np.matmul(), or the @ operator."],solution:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.
    """
    return np.dot(A, B)
    # Alternative: return A @ B
    # Alternative: return np.matmul(A, B)
`},{id:"numpy-broadcast-add",title:"Broadcasting Addition",section:"python-basics",difficulty:"medium",description:`
## Broadcasting Addition

Given a 2D matrix and a 1D vector, add the vector to each row of the matrix using broadcasting.

### Function Signature
\`\`\`python
def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Matrix has shape (m, n)
- Vector has shape (n,)
- 1 <= m, n <= 100

### Note
Broadcasting is a powerful NumPy feature that allows operations between arrays of different shapes.
    `,examples:[{input:"matrix = [[1, 2, 3], [4, 5, 6]], vector = [10, 20, 30]",output:"[[11, 22, 33], [14, 25, 36]]",explanation:"Vector is added to each row"}],starterCode:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.

    Args:
        matrix: 2D array of shape (m, n)
        vector: 1D array of shape (n,)

    Returns:
        Result matrix of shape (m, n)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic broadcasting",input:"([[1, 2, 3], [4, 5, 6]], [10, 20, 30])",expected:"[[11, 22, 33], [14, 25, 36]]",hidden:!1},{id:"2",description:"With zeros",input:"([[1, 2], [3, 4]], [0, 0])",expected:"[[1, 2], [3, 4]]",hidden:!1},{id:"3",description:"Negative values",input:"([[1, 2], [3, 4]], [-1, -2])",expected:"[[0, 0], [2, 2]]",hidden:!0}],hints:["NumPy automatically broadcasts the vector across rows.","Simply use the + operator - NumPy handles the rest!"],solution:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.
    """
    return matrix + vector
`}],tk=[{id:"normalize-features",title:"Normalize Features",section:"data-preprocessing",difficulty:"easy",description:`
## Normalize Features (Min-Max Scaling)

Implement min-max normalization to scale features to the range [0, 1].

### Formula
\`\`\`
X_normalized = (X - X_min) / (X_max - X_min)
\`\`\`

### Function Signature
\`\`\`python
def normalize(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array length: 2 <= n <= 1000
- All elements are finite numbers
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"[0.0, 0.25, 0.5, 0.75, 1.0]",explanation:"min=1, max=5, so (x-1)/(5-1) gives [0, 0.25, 0.5, 0.75, 1]"}],starterCode:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].

    Args:
        arr: Input array of numbers

    Returns:
        Normalized array with values in [0, 1]
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic normalization",input:"[1, 2, 3, 4, 5]",expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"2",description:"With negative values",input:"[-10, 0, 10]",expected:"[0.0, 0.5, 1.0]",hidden:!1},{id:"3",description:"Larger range",input:"[0, 50, 100]",expected:"[0.0, 0.5, 1.0]",hidden:!0}],hints:["Use np.min() and np.max() to find the range.","Apply the formula: (x - min) / (max - min)"],solution:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)
`},{id:"handle-missing-data",title:"Handle Missing Data",section:"data-preprocessing",difficulty:"medium",description:`
## Handle Missing Data

Replace NaN (missing) values in an array with the mean of non-NaN values.

### Function Signature
\`\`\`python
def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array contains at least one non-NaN value
- NaN values are represented as np.nan
    `,examples:[{input:"np.array([1.0, np.nan, 3.0, np.nan, 5.0])",output:"[1.0, 3.0, 3.0, 3.0, 5.0]",explanation:"Mean of [1, 3, 5] is 3, which replaces NaN values"}],starterCode:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.

    Args:
        arr: Input array that may contain NaN values

    Returns:
        Array with NaN values replaced by the mean
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic case",input:'[1.0, float("nan"), 3.0, float("nan"), 5.0]',expected:"[1.0, 3.0, 3.0, 3.0, 5.0]",hidden:!1},{id:"2",description:"Single NaN",input:'[2.0, 4.0, float("nan"), 6.0]',expected:"[2.0, 4.0, 4.0, 6.0]",hidden:!1},{id:"3",description:"NaN at start",input:'[float("nan"), 10.0, 20.0]',expected:"[15.0, 10.0, 20.0]",hidden:!0}],hints:["Use np.isnan() to find NaN values.","Use np.nanmean() to compute mean ignoring NaN values.","Use boolean indexing to replace values."],solution:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.
    """
    arr = arr.copy()  # Don't modify original
    mean_val = np.nanmean(arr)
    arr[np.isnan(arr)] = mean_val
    return arr
`},{id:"one-hot-encode",title:"One-Hot Encoding",section:"data-preprocessing",difficulty:"medium",description:`
## One-Hot Encoding

Convert an array of categorical integers to one-hot encoded format.

### Function Signature
\`\`\`python
def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
\`\`\`

### Example
For labels [0, 1, 2] with 3 classes:
\`\`\`
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
\`\`\`

### Constraints
- Labels are integers in range [0, num_classes - 1]
- 1 <= num_classes <= 100
    `,examples:[{input:"labels = [0, 1, 2, 1], num_classes = 3",output:"[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]",explanation:"Each label becomes a row with 1 at the label index"}],starterCode:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels: 1D array of integer labels
        num_classes: Total number of classes

    Returns:
        2D array of shape (len(labels), num_classes)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic encoding",input:"([0, 1, 2], 3)",expected:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",hidden:!1},{id:"2",description:"Repeated labels",input:"([0, 0, 1, 1], 2)",expected:"[[1, 0], [1, 0], [0, 1], [0, 1]]",hidden:!1},{id:"3",description:"More classes than used",input:"([0, 2], 4)",expected:"[[1, 0, 0, 0], [0, 0, 1, 0]]",hidden:!0}],hints:["Create a zeros matrix of shape (num_samples, num_classes).","Use array indexing to set the appropriate positions to 1.","np.eye() can be useful for creating one-hot encodings."],solution:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    """
    # Method 1: Using np.eye
    return np.eye(num_classes)[labels].astype(int)

    # Method 2: Manual approach
    # one_hot = np.zeros((len(labels), num_classes), dtype=int)
    # one_hot[np.arange(len(labels)), labels] = 1
    # return one_hot
`}],nk=[{id:"linear-regression-gd",title:"Linear Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Linear Regression with Gradient Descent

Implement simple linear regression using gradient descent optimization.

### Model
\`\`\`
y = w * x + b
\`\`\`

### Gradient Descent Updates
\`\`\`
w = w - learning_rate * dw
b = b - learning_rate * db
\`\`\`

Where:
- dw = (2/n) * sum((y_pred - y) * x)
- db = (2/n) * sum(y_pred - y)

### Function Signature
\`\`\`python
def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float, iterations: int) -> tuple:
\`\`\`

Returns (w, b) - the learned weight and bias.
    `,examples:[{input:"X = [1, 2, 3, 4], y = [2, 4, 6, 8], lr = 0.1, iters = 1000",output:"w ≈ 2.0, b ≈ 0.0",explanation:"The true relationship is y = 2x, so w should be close to 2"}],starterCode:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.

    Args:
        X: Input features (1D array)
        y: Target values (1D array)
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations

    Returns:
        Tuple of (weight, bias)
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    n = len(X)

    # Your gradient descent implementation here
    pass

    return (round(w, 2), round(b, 2))
`,testCases:[{id:"1",description:"Perfect linear relationship",input:"([1, 2, 3, 4], [2, 4, 6, 8], 0.1, 1000)",expected:"(2.0, 0.0)",hidden:!1},{id:"2",description:"With intercept",input:"([1, 2, 3, 4], [3, 5, 7, 9], 0.1, 1000)",expected:"(2.0, 1.0)",hidden:!1},{id:"3",description:"Different slope",input:"([0, 1, 2, 3], [1, 4, 7, 10], 0.1, 1000)",expected:"(3.0, 1.0)",hidden:!0}],hints:["First compute predictions: y_pred = w * X + b","Then compute gradients: dw = (2/n) * sum((y_pred - y) * X)","Update weights: w = w - learning_rate * dw"],solution:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.
    """
    w = 0.0
    b = 0.0
    n = len(X)

    for _ in range(iterations):
        # Forward pass
        y_pred = w * X + b

        # Compute gradients
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return (round(w, 2), round(b, 2))
`},{id:"logistic-regression",title:"Sigmoid Function",section:"supervised-learning",difficulty:"easy",description:`
## Sigmoid Function

Implement the sigmoid activation function used in logistic regression.

### Formula
\`\`\`
sigmoid(x) = 1 / (1 + exp(-x))
\`\`\`

### Properties
- Output is always between 0 and 1
- sigmoid(0) = 0.5
- Monotonically increasing

### Function Signature
\`\`\`python
def sigmoid(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([0, 1, -1])",output:"[0.5, 0.731059, 0.268941]",explanation:"sigmoid(0)=0.5, sigmoid(1)≈0.73, sigmoid(-1)≈0.27"}],starterCode:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with sigmoid applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero input",input:"[0]",expected:"[0.5]",hidden:!1},{id:"2",description:"Positive and negative",input:"[-1, 0, 1]",expected:"[0.268941, 0.5, 0.731059]",hidden:!1},{id:"3",description:"Large values",input:"[-10, 10]",expected:"[0.000045, 0.999955]",hidden:!0}],hints:["Use np.exp() for the exponential function.","The formula is 1 / (1 + exp(-x))"],solution:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))
`},{id:"decision-tree-split",title:"Gini Impurity",section:"supervised-learning",difficulty:"medium",description:`
## Gini Impurity

Calculate the Gini impurity for a set of labels. This metric is used in decision trees to determine the best split.

### Formula
\`\`\`
Gini = 1 - sum(p_i^2)
\`\`\`

Where p_i is the proportion of class i in the set.

### Properties
- Gini = 0 means pure (all same class)
- Gini = 0.5 means maximum impurity (for binary classification with equal split)

### Function Signature
\`\`\`python
def gini_impurity(labels: np.ndarray) -> float:
\`\`\`
    `,examples:[{input:"np.array([0, 0, 0, 0])",output:"0.0",explanation:"All same class, pure node"},{input:"np.array([0, 0, 1, 1])",output:"0.5",explanation:"Equal split, maximum impurity for binary"}],starterCode:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: Array of class labels (integers)

    Returns:
        Gini impurity value between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Pure node",input:"[0, 0, 0, 0]",expected:"0.0",hidden:!1},{id:"2",description:"Maximum impurity",input:"[0, 0, 1, 1]",expected:"0.5",hidden:!1},{id:"3",description:"Unequal split",input:"[0, 0, 0, 1]",expected:"0.375",hidden:!0}],hints:["Count the occurrences of each class.","Calculate the proportion (probability) of each class.","Gini = 1 - sum of squared probabilities."],solution:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.
    """
    if len(labels) == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(labels)

    # Gini = 1 - sum(p^2)
    return 1 - np.sum(probabilities ** 2)
`},{id:"logistic-regression-full",title:"Logistic Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Logistic Regression with Gradient Descent

Implement binary logistic regression from scratch.

### Model
\`\`\`
z = X @ w + b
y_pred = sigmoid(z) = 1 / (1 + exp(-z))
\`\`\`

### Loss (Binary Cross-Entropy)
\`\`\`
L = -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
\`\`\`

### Gradients
\`\`\`
dw = 1/m * X.T @ (y_pred - y)
db = 1/m * sum(y_pred - y)
\`\`\`
    `,examples:[{input:"X (100, 2), y binary labels, 1000 iterations",output:"Trained weights and bias",explanation:"Learns decision boundary separating classes"}],starterCode:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: Features (m samples, n features)
        y: Binary labels (m,)
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        w: Learned weights (n,)
        b: Learned bias (scalar)
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    # Your code here
    pass

    return np.round(w, 4), round(b, 4)
`,testCases:[{id:"1",description:"Simple separable data",input:"(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), 0.5, 1000)",expected:"converged",hidden:!1},{id:"2",description:"Predictions reasonable",input:"prediction_test",expected:"True",hidden:!0}],hints:["Forward: z = X @ w + b, then y_pred = sigmoid(z)","Gradients: dw = (1/m) * X.T @ (y_pred - y)","Update: w = w - lr * dw, b = b - lr * db"],solution:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        # Forward pass
        z = X @ w + b
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return np.round(w, 4), round(b, 4)
`},{id:"binary-cross-entropy",title:"Binary Cross-Entropy Loss",section:"supervised-learning",difficulty:"easy",description:`
## Binary Cross-Entropy Loss

Implement the binary cross-entropy (log loss) function.

### Formula
\`\`\`
BCE = -1/m * sum(y * log(p) + (1-y) * log(1-p))
\`\`\`

Where:
- y: True labels (0 or 1)
- p: Predicted probabilities
- m: Number of samples

### Numerical Stability
Clip predictions to avoid log(0):
\`\`\`python
p = np.clip(p, 1e-15, 1 - 1e-15)
\`\`\`
    `,examples:[{input:"y = [1, 0, 1], p = [0.9, 0.1, 0.8]",output:"0.1643",explanation:"Low loss for confident correct predictions"}],starterCode:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities

    Returns:
        loss: Scalar BCE loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect predictions",input:"([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"([1, 0, 1], [0.9, 0.1, 0.8])",expected:"0.1643",hidden:!1}],hints:["Clip predictions for numerical stability","Apply the formula element-wise","Take the mean over all samples"],solution:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip for numerical stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # BCE formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return round(loss, 4)
`}],rk=[{id:"kmeans-clustering",title:"K-Means: Assign to Nearest Centroid",section:"unsupervised-learning",difficulty:"medium",description:`
## K-Means: Assign Points to Nearest Centroid

Implement the assignment step of K-means clustering: given data points and centroids, assign each point to its nearest centroid.

### Function Signature
\`\`\`python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data points of shape (n_samples, n_features)
- centroids: Cluster centers of shape (k, n_features)

### Returns
- Array of cluster assignments (integers 0 to k-1)

### Distance
Use Euclidean distance: sqrt(sum((x - c)^2))
    `,examples:[{input:"X = [[0, 0], [1, 1], [10, 10]], centroids = [[0, 0], [10, 10]]",output:"[0, 0, 1]",explanation:"First two points closer to centroid 0, last point closer to centroid 1"}],starterCode:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.

    Args:
        X: Data points of shape (n_samples, n_features)
        centroids: Cluster centers of shape (k, n_features)

    Returns:
        Array of cluster assignments (shape: n_samples)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2D case",input:"([[0, 0], [1, 1], [10, 10]], [[0, 0], [10, 10]])",expected:"[0, 0, 1]",hidden:!1},{id:"2",description:"Three clusters",input:"([[0, 0], [5, 5], [10, 10]], [[0, 0], [5, 5], [10, 10]])",expected:"[0, 1, 2]",hidden:!1},{id:"3",description:"Tiebreaker (first centroid wins)",input:"([[5, 5]], [[0, 0], [10, 10]])",expected:"[0]",hidden:!0}],hints:["Calculate distance from each point to each centroid.","Use np.linalg.norm() for Euclidean distance.","Use np.argmin() to find the closest centroid."],solution:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.
    """
    # Calculate distances from each point to each centroid
    # Using broadcasting: X[:, np.newaxis] has shape (n, 1, d)
    # centroids has shape (k, d)
    # Difference has shape (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Return index of minimum distance for each point
    return np.argmin(distances, axis=1)
`},{id:"pca-implementation",title:"PCA: Center Data",section:"unsupervised-learning",difficulty:"easy",description:`
## PCA: Center the Data

The first step in PCA is to center the data by subtracting the mean of each feature.

### Function Signature
\`\`\`python
def center_data(X: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data matrix of shape (n_samples, n_features)

### Returns
- Centered data where each column has mean 0

### Why Center?
Centering ensures PCA finds directions of maximum variance from the origin, not from an arbitrary point.
    `,examples:[{input:"X = [[1, 2], [3, 4], [5, 6]]",output:"[[-2, -2], [0, 0], [2, 2]]",explanation:"Column means are [3, 4], so we subtract them"}],starterCode:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Centered data matrix of same shape
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic centering",input:"[[1, 2], [3, 4], [5, 6]]",expected:"[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]",hidden:!1},{id:"2",description:"Already centered",input:"[[-1, -1], [0, 0], [1, 1]]",expected:"[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]",hidden:!1},{id:"3",description:"Single column",input:"[[10], [20], [30]]",expected:"[[-10.0], [0.0], [10.0]]",hidden:!0}],hints:["Calculate the mean of each column using np.mean() with axis=0.","Subtract the mean from the data (broadcasting handles this)."],solution:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.
    """
    mean = np.mean(X, axis=0)
    return X - mean
`}],ik=[{id:"perceptron",title:"ReLU Activation",section:"deep-learning",difficulty:"easy",description:`
## ReLU Activation Function

Implement the Rectified Linear Unit (ReLU) activation function, one of the most popular activation functions in deep learning.

### Formula
\`\`\`
ReLU(x) = max(0, x)
\`\`\`

### Properties
- Output is x if x > 0, else 0
- Helps with vanishing gradient problem
- Computationally efficient

### Function Signature
\`\`\`python
def relu(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([-2, -1, 0, 1, 2])",output:"[0, 0, 0, 1, 2]",explanation:"Negative values become 0, positive values unchanged"}],starterCode:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array with ReLU applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed values",input:"[-2, -1, 0, 1, 2]",expected:"[0, 0, 0, 1, 2]",hidden:!1},{id:"2",description:"All negative",input:"[-5, -3, -1]",expected:"[0, 0, 0]",hidden:!1},{id:"3",description:"All positive",input:"[1, 2, 3]",expected:"[1, 2, 3]",hidden:!0}],hints:["Use np.maximum() to compare with 0.","Alternatively, use boolean indexing or np.where()."],solution:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.
    """
    return np.maximum(0, x)
`},{id:"neural-network-forward",title:"Dense Layer Forward Pass",section:"deep-learning",difficulty:"medium",description:`
## Dense Layer Forward Pass

Implement the forward pass of a fully-connected (dense) neural network layer.

### Formula
\`\`\`
output = activation(X @ W + b)
\`\`\`

Where:
- X: Input of shape (batch_size, input_dim)
- W: Weights of shape (input_dim, output_dim)
- b: Bias of shape (output_dim,)

### Function Signature
\`\`\`python
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
\`\`\`

Use ReLU as the activation function.
    `,examples:[{input:"X = [[1, 2]], W = [[1, 0], [0, 1]], b = [1, 1]",output:"[[2, 3]]",explanation:"X @ W = [[1, 2]], + b = [[2, 3]], ReLU keeps positive"}],starterCode:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.

    Args:
        X: Input of shape (batch_size, input_dim)
        W: Weights of shape (input_dim, output_dim)
        b: Bias of shape (output_dim,)

    Returns:
        Output of shape (batch_size, output_dim)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic forward pass",input:"([[1, 2]], [[1, 0], [0, 1]], [1, 1])",expected:"[[2, 3]]",hidden:!1},{id:"2",description:"With negative pre-activation",input:"([[1, 1]], [[1, -2], [-1, 1]], [0, 0])",expected:"[[0, 0]]",hidden:!1},{id:"3",description:"Batch of 2",input:"([[1, 0], [0, 1]], [[2, 0], [0, 2]], [0, 0])",expected:"[[2, 0], [0, 2]]",hidden:!0}],hints:["First compute the linear transformation: Z = X @ W + b","Then apply ReLU: output = max(0, Z)"],solution:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.
    """
    # Linear transformation
    Z = np.dot(X, W) + b
    # ReLU activation
    return np.maximum(0, Z)
`},{id:"backpropagation",title:"Softmax Function",section:"deep-learning",difficulty:"medium",description:`
## Softmax Function

Implement the softmax function, used to convert raw scores to probabilities in multi-class classification.

### Formula
\`\`\`
softmax(x)_i = exp(x_i) / sum(exp(x_j))
\`\`\`

### Properties
- Output sums to 1
- All outputs are positive
- Larger inputs get larger probabilities

### Function Signature
\`\`\`python
def softmax(x: np.ndarray) -> np.ndarray:
\`\`\`

**Note**: For numerical stability, subtract max(x) before computing exp.
    `,examples:[{input:"np.array([1, 2, 3])",output:"[0.090031, 0.244728, 0.665241]",explanation:"Higher values get higher probabilities, sum = 1"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input array (1D)

    Returns:
        Array of same shape with softmax probabilities
    """
    # Your code here
    # Hint: subtract max for numerical stability
    pass
`,testCases:[{id:"1",description:"Basic softmax",input:"[1, 2, 3]",expected:"[0.090031, 0.244728, 0.665241]",hidden:!1},{id:"2",description:"Equal inputs",input:"[1, 1, 1]",expected:"[0.333333, 0.333333, 0.333333]",hidden:!1},{id:"3",description:"Large values",input:"[100, 101, 102]",expected:"[0.090031, 0.244728, 0.665241]",hidden:!0}],hints:["Subtract the max value for numerical stability.","Compute exp of each element.","Divide by the sum of all exp values."],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
`}],ak=[{id:"precision-recall-f1",title:"Precision and Recall",section:"model-evaluation",difficulty:"medium",description:`
## Precision and Recall

Calculate precision and recall from true labels and predictions.

### Formulas
\`\`\`
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
\`\`\`

Where:
- TP = True Positives (predicted 1, actual 1)
- FP = False Positives (predicted 1, actual 0)
- FN = False Negatives (predicted 0, actual 1)

### Function Signature
\`\`\`python
def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
\`\`\`

Returns (precision, recall) rounded to 4 decimal places.
    `,examples:[{input:"y_true = [1, 1, 0, 1, 0], y_pred = [1, 0, 0, 1, 1]",output:"(0.6667, 0.6667)",explanation:"TP=2, FP=1, FN=1. Precision=2/3, Recall=2/3"}],starterCode:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Tuple of (precision, recall) rounded to 4 decimal places
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed predictions",input:"([1, 1, 0, 1, 0], [1, 0, 0, 1, 1])",expected:"(0.6667, 0.6667)",hidden:!1},{id:"2",description:"Perfect predictions",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"(1.0, 1.0)",hidden:!1},{id:"3",description:"High precision, low recall",input:"([1, 1, 1, 1, 0], [1, 0, 0, 0, 0])",expected:"(1.0, 0.25)",hidden:!0}],hints:["Count TP where both y_true and y_pred are 1.","Count FP where y_pred is 1 but y_true is 0.","Count FN where y_true is 1 but y_pred is 0."],solution:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return (round(precision, 4), round(recall, 4))
`},{id:"cross-validation",title:"K-Fold Split Indices",section:"model-evaluation",difficulty:"medium",description:`
## K-Fold Cross-Validation Indices

Generate train and validation indices for K-fold cross-validation.

### Concept
Split data into K equal parts. For each fold:
- Use 1 part for validation
- Use remaining K-1 parts for training

### Function Signature
\`\`\`python
def kfold_indices(n_samples: int, k: int) -> list:
\`\`\`

Returns list of (train_indices, val_indices) tuples for each fold.
    `,examples:[{input:"n_samples = 6, k = 3",output:"[([2,3,4,5], [0,1]), ([0,1,4,5], [2,3]), ([0,1,2,3], [4,5])]",explanation:"Split [0,1,2,3,4,5] into 3 folds of size 2"}],starterCode:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        n_samples: Total number of samples
        k: Number of folds

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic 3-fold",input:"(6, 3)",expected:"[([2, 3, 4, 5], [0, 1]), ([0, 1, 4, 5], [2, 3]), ([0, 1, 2, 3], [4, 5])]",hidden:!1},{id:"2",description:"2-fold",input:"(4, 2)",expected:"[([2, 3], [0, 1]), ([0, 1], [2, 3])]",hidden:!1},{id:"3",description:"4-fold",input:"(8, 4)",expected:"[([2, 3, 4, 5, 6, 7], [0, 1]), ([0, 1, 4, 5, 6, 7], [2, 3]), ([0, 1, 2, 3, 6, 7], [4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7])]",hidden:!0}],hints:["Divide indices into k equal-sized chunks.","For each fold, one chunk is validation, rest are training.","Use np.array_split() to create chunks."],solution:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = list(range(current, current + fold_size))
        train_indices = [i for i in range(n_samples) if i not in val_indices]
        folds.append((train_indices, val_indices))
        current += fold_size

    return folds
`},{id:"confusion-matrix",title:"Accuracy Score",section:"model-evaluation",difficulty:"easy",description:`
## Accuracy Score

Calculate the accuracy of predictions - the proportion of correct predictions.

### Formula
\`\`\`
Accuracy = (Number of correct predictions) / (Total predictions)
\`\`\`

### Function Signature
\`\`\`python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
\`\`\`

### Note
While simple, accuracy can be misleading for imbalanced datasets.
    `,examples:[{input:"y_true = [0, 1, 1, 0, 1], y_pred = [0, 1, 0, 0, 1]",output:"0.8",explanation:"4 correct out of 5 predictions"}],starterCode:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic accuracy",input:"([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])",expected:"0.8",hidden:!1},{id:"2",description:"Perfect accuracy",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"1.0",hidden:!1},{id:"3",description:"Zero accuracy",input:"([0, 0, 1, 1], [1, 1, 0, 0])",expected:"0.0",hidden:!0}],hints:["Compare y_true and y_pred element-wise.","Count matches and divide by total.","np.mean() of boolean array gives proportion of True."],solution:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
`}],ok=[{id:"mlp-forward",title:"MLP Forward Pass",section:"neural-networks",difficulty:"medium",description:`
## Multi-Layer Perceptron Forward Pass

Implement the forward pass for a 2-layer MLP (Multi-Layer Perceptron).

### Architecture
\`\`\`
Input (n_features) → Hidden (n_hidden) → Output (n_classes)
\`\`\`

### Forward Pass
\`\`\`
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
output = softmax(Z2)
\`\`\`

### Function Signature
\`\`\`python
def mlp_forward(X, W1, b1, W2, b2):
    # Returns: output probabilities, cache (for backprop)
\`\`\`
    `,examples:[{input:"X shape (2, 3), hidden_size=4, output_size=2",output:"Probabilities shape (2, 2), each row sums to 1",explanation:"Batch of 2 samples through MLP to 2 classes"}],starterCode:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass for a 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        W1: First layer weights (n_features, n_hidden)
        b1: First layer bias (n_hidden,)
        W2: Second layer weights (n_hidden, n_classes)
        b2: Second layer bias (n_classes,)

    Returns:
        output: Class probabilities (batch_size, n_classes)
        cache: Tuple of intermediate values for backprop
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape correct",input:"(np.array([[1, 2, 3], [4, 5, 6]]), np.ones((3, 4)), np.zeros(4), np.ones((4, 2)), np.zeros(2))",expected:"(2, 2)",hidden:!1},{id:"2",description:"Probabilities sum to 1",input:"(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.zeros(2), np.array([[1, 0], [0, 1]]), np.zeros(2))",expected:"1.0",hidden:!1}],hints:["First compute Z1 = X @ W1 + b1","Apply ReLU to get A1","Then Z2 = A1 @ W2 + b2","Apply softmax to get output probabilities"],solution:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    # First layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = A1 @ W2 + b2
    output = softmax(Z2)

    # Cache for backprop
    cache = (X, Z1, A1, Z2, W1, W2)

    return output, cache
`},{id:"backprop-gradients",title:"Backpropagation Gradients",section:"neural-networks",difficulty:"hard",description:`
## Backpropagation: Compute Gradients

Implement backpropagation to compute gradients for a 2-layer MLP with cross-entropy loss.

### Chain Rule
For each layer, we compute:
\`\`\`
dL/dW = dL/dZ * dZ/dW
dL/db = dL/dZ * dZ/db
\`\`\`

### Gradient Flow
\`\`\`
Loss → dZ2 → dW2, db2 → dA1 → dZ1 (ReLU) → dW1, db1
\`\`\`

### Key Formulas
- dZ2 = output - Y_onehot (softmax + cross-entropy)
- dW2 = A1.T @ dZ2
- dZ1 = (dZ2 @ W2.T) * (Z1 > 0)  (ReLU derivative)
- dW1 = X.T @ dZ1
    `,examples:[{input:"Predictions vs true labels",output:"Gradients dW1, db1, dW2, db2",explanation:"Gradients flow backwards through the network"}],starterCode:`import numpy as np

def backprop(X, Y, output, cache):
    """
    Compute gradients for 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        Y: One-hot encoded labels (batch_size, n_classes)
        output: Predicted probabilities from forward pass
        cache: Cached values from forward pass (X, Z1, A1, Z2, W1, W2)

    Returns:
        grads: Dictionary with dW1, db1, dW2, db2
    """
    m = X.shape[0]  # batch size
    X, Z1, A1, Z2, W1, W2 = cache

    # Your code here
    pass
`,testCases:[{id:"1",description:"dW2 shape matches W2",input:"shapes_match",expected:"True",hidden:!1},{id:"2",description:"Gradients are non-zero",input:"non_zero_grads",expected:"True",hidden:!0}],hints:["Start from the output: dZ2 = output - Y","dW2 = (1/m) * A1.T @ dZ2","Propagate through ReLU: dZ1 = (dZ2 @ W2.T) * (Z1 > 0)","dW1 = (1/m) * X.T @ dZ1"],solution:`import numpy as np

def backprop(X, Y, output, cache):
    m = X.shape[0]
    X, Z1, A1, Z2, W1, W2 = cache

    # Output layer gradient (softmax + cross-entropy)
    dZ2 = output - Y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0)

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
`},{id:"cross-entropy-loss",title:"Cross-Entropy Loss",section:"neural-networks",difficulty:"easy",description:`
## Cross-Entropy Loss

Implement the cross-entropy loss function for multi-class classification.

### Formula
\`\`\`
L = -1/m * sum(Y * log(Y_pred))
\`\`\`

Where:
- Y is one-hot encoded true labels
- Y_pred is predicted probabilities
- m is batch size

### Numerical Stability
Add small epsilon to avoid log(0):
\`\`\`python
np.log(Y_pred + 1e-15)
\`\`\`
    `,examples:[{input:"Y_pred = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], Y = [[1,0,0], [0,1,0]]",output:"0.2231",explanation:"-1/2 * (log(0.7) + log(0.8))"}],starterCode:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    """
    Compute cross-entropy loss.

    Args:
        Y_pred: Predicted probabilities (batch_size, n_classes)
        Y_true: One-hot encoded true labels (batch_size, n_classes)

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect prediction",input:"([[1.0, 0.0], [0.0, 1.0]], [[1, 0], [0, 1]])",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"([[0.7, 0.3], [0.2, 0.8]], [[1, 0], [0, 1]])",expected:"0.2899",hidden:!1}],hints:["Use np.log with a small epsilon for numerical stability","Multiply element-wise: Y_true * np.log(Y_pred)","Sum and negate, then divide by batch size"],solution:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_pred.shape[0]
    epsilon = 1e-15
    log_probs = np.log(Y_pred + epsilon)
    loss = -np.sum(Y_true * log_probs) / m
    return round(loss, 4)
`},{id:"weight-init",title:"Xavier/He Weight Initialization",section:"neural-networks",difficulty:"medium",description:`
## Weight Initialization

Implement Xavier and He weight initialization to prevent vanishing/exploding gradients.

### Xavier Initialization (for tanh/sigmoid)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / (n_in + n_out))
\`\`\`

### He Initialization (for ReLU)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / n_in)
\`\`\`

### Why It Matters
- Keeps variance of activations stable across layers
- Prevents gradients from vanishing or exploding
- Enables training of deep networks
    `,examples:[{input:'n_in=784, n_out=256, method="he"',output:"Weights with std ≈ 0.0505",explanation:"sqrt(2/784) ≈ 0.0505"}],starterCode:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    """
    Initialize weights using Xavier or He initialization.

    Args:
        n_in: Number of input units
        n_out: Number of output units
        method: 'xavier' or 'he'

    Returns:
        W: Initialized weight matrix (n_in, n_out)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Xavier std correct",input:'(100, 50, "xavier")',expected:"0.1155",hidden:!1},{id:"2",description:"He std correct",input:'(100, 50, "he")',expected:"0.1414",hidden:!1}],hints:["Use np.random.randn to generate random numbers","Xavier: multiply by sqrt(2 / (n_in + n_out))","He: multiply by sqrt(2 / n_in)"],solution:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    np.random.seed(42)

    if method == 'xavier':
        std = np.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = np.sqrt(2.0 / n_in)
    else:
        raise ValueError("Method must be 'xavier' or 'he'")

    W = np.random.randn(n_in, n_out) * std
    return round(np.std(W), 4)
`},{id:"batch-norm",title:"Batch Normalization",section:"neural-networks",difficulty:"medium",description:`
## Batch Normalization

Implement batch normalization to stabilize training.

### Forward Pass
\`\`\`
1. Compute batch mean: μ = mean(x)
2. Compute batch variance: σ² = var(x)
3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
4. Scale and shift: out = γ * x_norm + β
\`\`\`

### Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
    `,examples:[{input:"X with mean=5, var=4, gamma=1, beta=0",output:"Normalized X with mean≈0, var≈1",explanation:"BatchNorm normalizes each feature"}],starterCode:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        X: Input (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized output
        cache: Values needed for backward pass
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"(np.array([[1, 2], [3, 4], [5, 6]]), np.ones(2), np.zeros(2))",expected:"True",hidden:!1},{id:"2",description:"Gamma scales output",input:"(np.array([[0, 0], [2, 2]]), np.array([2, 3]), np.zeros(2))",expected:"[2.0, 3.0]",hidden:!0}],hints:["Compute mean along axis=0 (batch dimension)","Compute variance along axis=0","Normalize: (X - mean) / sqrt(var + eps)","Apply scale and shift: gamma * normalized + beta"],solution:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # Compute batch statistics
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    # Normalize
    X_norm = (X - mu) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * X_norm + beta

    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, eps)

    return out, cache
`},{id:"dropout",title:"Dropout",section:"neural-networks",difficulty:"easy",description:`
## Dropout Regularization

Implement dropout to prevent overfitting.

### Training Mode
\`\`\`
1. Generate mask: mask = (rand < keep_prob) / keep_prob
2. Apply mask: out = x * mask
\`\`\`

### Key Points
- Randomly "drop" neurons with probability (1 - keep_prob)
- Scale by 1/keep_prob to maintain expected value
- Disabled during inference (return input unchanged)
    `,examples:[{input:"X = [[1, 2, 3, 4]], keep_prob=0.5",output:"Some values zeroed, others scaled by 2",explanation:"Half neurons dropped, rest doubled"}],starterCode:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    """
    Apply dropout to input.

    Args:
        X: Input array
        keep_prob: Probability of keeping a neuron
        training: Whether in training mode

    Returns:
        out: Output after dropout
        mask: Dropout mask (for backward pass)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Inference returns unchanged",input:"(np.array([[1, 2, 3, 4]]), 0.5, False)",expected:"[[1, 2, 3, 4]]",hidden:!1},{id:"2",description:"Expected value preserved",input:"expected_value_test",expected:"True",hidden:!0}],hints:["In training: create binary mask with np.random.rand","Scale mask by 1/keep_prob (inverted dropout)","In inference: return input unchanged"],solution:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    if not training:
        return X, None

    # Create mask and scale
    mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
    out = X * mask

    return out, mask
`}],sk=[{id:"conv2d-forward",title:"2D Convolution",section:"cnn",difficulty:"hard",description:`
## 2D Convolution Operation

Implement the forward pass of a 2D convolution (no padding, stride=1).

### Operation
For each position (i, j) in the output:
\`\`\`
out[i, j] = sum(input[i:i+kH, j:j+kW] * kernel)
\`\`\`

### Output Size
\`\`\`
out_height = input_height - kernel_height + 1
out_width = input_width - kernel_width + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d(image, kernel):
    # image: (H, W)
    # kernel: (kH, kW)
    # output: (H-kH+1, W-kW+1)
\`\`\`
    `,examples:[{input:"image 4x4, kernel 3x3",output:"output 2x2",explanation:"4-3+1 = 2 in each dimension"}],starterCode:`import numpy as np

def conv2d(image, kernel):
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        output: Convolved image (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Identity kernel",input:"([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])",expected:"[[5]]",hidden:!1},{id:"2",description:"Edge detection",input:"([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])",expected:"[[4, 4], [4, 4]]",hidden:!1}],hints:["Use nested loops to slide the kernel over the image","At each position, compute element-wise product and sum","Output size is (H-kH+1, W-kW+1)"],solution:`import numpy as np

def conv2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output.astype(int).tolist()
`},{id:"max-pool",title:"Max Pooling",section:"cnn",difficulty:"medium",description:`
## Max Pooling

Implement 2x2 max pooling with stride 2.

### Operation
Divide input into non-overlapping 2x2 regions and take maximum of each.

### Output Size
\`\`\`
out_height = input_height // 2
out_width = input_width // 2
\`\`\`

### Purpose
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation
    `,examples:[{input:"[[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]",output:"[[6, 8], [14, 16]]",explanation:"Max of each 2x2 region"}],starterCode:`import numpy as np

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling.

    Args:
        image: Input image (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"4x4 to 2x2",input:"[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",expected:"[[6, 8], [14, 16]]",hidden:!1},{id:"2",description:"With negative values",input:"[[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12], [-13, -14, -15, -16]]",expected:"[[-1, -3], [-9, -11]]",hidden:!0}],hints:["Iterate with step size = pool_size","For each 2x2 region, use np.max()","Output dimensions are input dimensions // pool_size"],solution:`import numpy as np

def max_pool2d(image, pool_size=2):
    image = np.array(image)
    H, W = image.shape

    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output.astype(int).tolist()
`},{id:"flatten-layer",title:"Flatten Layer",section:"cnn",difficulty:"easy",description:`
## Flatten Layer

Implement the flatten operation that converts a 3D feature map to a 1D vector for the fully connected layer.

### Operation
\`\`\`
(batch, height, width, channels) → (batch, height * width * channels)
\`\`\`

### Usage
- Connects convolutional layers to fully connected layers
- Preserves batch dimension
    `,examples:[{input:"shape (2, 4, 4, 3)",output:"shape (2, 48)",explanation:"4 * 4 * 3 = 48 features per sample"}],starterCode:`import numpy as np

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct output shape",input:"np.random.randn(2, 4, 4, 3)",expected:"(2, 48)",hidden:!1},{id:"2",description:"Values preserved",input:"np.arange(24).reshape(1, 2, 3, 4)",expected:"True",hidden:!0}],hints:["Get batch size as X.shape[0]","Use reshape to flatten remaining dimensions","np.reshape(X, (batch_size, -1)) uses -1 to infer size"],solution:`import numpy as np

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`},{id:"conv-output-size",title:"Convolution Output Size",section:"cnn",difficulty:"easy",description:`
## Calculate Convolution Output Size

Implement a function to calculate the output dimensions of a convolution layer.

### Formula
\`\`\`
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
\`\`\`

### Parameters
- **input_size**: Height or width of input
- **kernel_size**: Height or width of kernel
- **padding**: Zero-padding added to input
- **stride**: Step size of kernel
    `,examples:[{input:"input=32, kernel=3, padding=1, stride=1",output:"32",explanation:"(32 - 3 + 2*1) / 1 + 1 = 32"}],starterCode:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same padding",input:"(32, 3, 1, 1)",expected:"32",hidden:!1},{id:"2",description:"No padding, stride 2",input:"(32, 3, 0, 2)",expected:"15",hidden:!1},{id:"3",description:"Large kernel",input:"(224, 7, 3, 2)",expected:"112",hidden:!0}],hints:["Apply the formula: (input - kernel + 2*padding) / stride + 1","Use integer division (//) for the result"],solution:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`},{id:"conv2d-advanced",title:"Advanced 2D Convolution",section:"cnn",difficulty:"hard",description:`
## Advanced 2D Convolution

Implement a full-featured 2D convolution with **padding**, **stride**, and **groups** support.

### Parameters
- **padding**: Zero-padding added to input borders
- **stride**: Step size when sliding the kernel
- **groups**: Split input/output channels into groups (used in depthwise separable convolutions)

### Group Convolution
When \`groups > 1\`:
- Input channels are split into \`groups\` chunks
- Output channels are split into \`groups\` chunks
- Each group's input is convolved only with its corresponding kernel group
- \`in_channels\` and \`out_channels\` must be divisible by \`groups\`

### Special Cases
- \`groups=1\`: Standard convolution
- \`groups=in_channels=out_channels\`: Depthwise convolution (MobileNet)

### Output Size Formula
\`\`\`
H_out = (H_in + 2*padding - kernel_height) // stride + 1
W_out = (W_in + 2*padding - kernel_width) // stride + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    # input: (batch, in_channels, H, W)
    # kernel: (out_channels, in_channels//groups, kH, kW)
    # output: (batch, out_channels, H_out, W_out)
\`\`\`
    `,examples:[{input:"input (1, 4, 5, 5), kernel (8, 2, 3, 3), groups=2",output:"output (1, 8, 3, 3)",explanation:"Groups=2: channels 0-1 use kernels 0-3, channels 2-3 use kernels 4-7"},{input:"input (1, 1, 5, 5), kernel (1, 1, 3, 3), padding=1, stride=2",output:"output (1, 1, 3, 3)",explanation:"(5+2*1-3)//2+1 = 3"}],starterCode:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic convolution with padding and stride",input:"(np.ones((1, 1, 5, 5)), np.ones((1, 1, 3, 3)), 1, 2, 1)",expected:"(1, 1, 3, 3)",hidden:!1},{id:"2",description:"Grouped convolution (groups=2)",input:"(np.arange(32).reshape(1, 2, 4, 4).astype(float), np.ones((4, 1, 2, 2)), 0, 1, 2)",expected:"(1, 4, 3, 3)",hidden:!1},{id:"3",description:"Depthwise convolution (groups=in_channels)",input:"(np.ones((2, 3, 4, 4)), np.ones((3, 1, 2, 2)), 0, 1, 3)",expected:"(2, 3, 3, 3)",hidden:!0},{id:"4",description:"Verify correct output values with groups",input:"verify_groups_output",expected:"True",hidden:!0}],hints:["First, pad the input using np.pad() along the H and W dimensions","Calculate output dimensions: H_out = (H_padded - kH) // stride + 1","For groups: split input channels into chunks of size in_channels//groups","For groups: split kernels into chunks of size out_channels//groups","Convolve each input group with its corresponding kernel group","Concatenate results along the channel dimension"],solution:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    input = np.array(input, dtype=float)
    kernel = np.array(kernel, dtype=float)

    batch, in_channels, H, W = input.shape
    out_channels, kernel_in_channels, kH, kW = kernel.shape

    # Pad input
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    _, _, H_padded, W_padded = input.shape

    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # Initialize output
    output = np.zeros((batch, out_channels, H_out, W_out))

    # Channels per group
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    for g in range(groups):
        # Input channels for this group
        in_start = g * in_channels_per_group
        in_end = in_start + in_channels_per_group

        # Output channels for this group
        out_start = g * out_channels_per_group
        out_end = out_start + out_channels_per_group

        # Get the input slice for this group
        input_group = input[:, in_start:in_end, :, :]

        # Get the kernels for this group
        kernel_group = kernel[out_start:out_end, :, :, :]

        # Perform convolution for this group
        for b in range(batch):
            for oc in range(out_channels_per_group):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = input_group[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, out_start + oc, i, j] = np.sum(region * kernel_group[oc])

    return output
`}],lk=[{id:"scaled-dot-product-attention",title:"Scaled Dot-Product Attention",section:"transformers",difficulty:"medium",description:`
## Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

### Formula
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

Where:
- Q: Queries (seq_len, d_k)
- K: Keys (seq_len, d_k)
- V: Values (seq_len, d_v)
- d_k: Dimension of keys

### Steps
1. Compute attention scores: Q @ K.T
2. Scale by sqrt(d_k)
3. Apply softmax
4. Multiply by V
    `,examples:[{input:"Q, K, V all (4, 8) - 4 tokens, 8 dims",output:"output (4, 8)",explanation:"Each token attends to all tokens"}],starterCode:`import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape correct",input:"(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))",expected:"(4, 8)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"attention_sum_test",expected:"True",hidden:!1}],hints:["Compute scores = Q @ K.T","Scale by sqrt(d_k) where d_k = Q.shape[-1]","Apply softmax along the last axis","Return softmax(scores) @ V"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scores)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights
`},{id:"multi-head-attention",title:"Multi-Head Attention",section:"transformers",difficulty:"hard",description:`
## Multi-Head Attention

Implement multi-head attention from the Transformer architecture.

### Concept
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
\`\`\`

### Steps
1. Project Q, K, V with different learned projections for each head
2. Apply scaled dot-product attention in parallel
3. Concatenate heads
4. Project back to original dimension
    `,examples:[{input:"X (4, 64), num_heads=8, d_model=64",output:"output (4, 64)",explanation:"8 heads with d_k=8 each, concatenated to 64"}],starterCode:`import numpy as np

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism.

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"(np.random.randn(4, 64), np.random.randn(4, 64), np.random.randn(4, 64), 8)",expected:"(4, 64)",hidden:!1}],hints:["Reshape Q, K, V to (seq_len, num_heads, d_k)","Apply attention for each head","Reshape back to (seq_len, d_model)"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention for each head
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention = softmax(scores)
    heads = attention @ V

    # Transpose and reshape back
    heads = heads.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
    output = heads.reshape(seq_len, d_model)

    return output
`},{id:"positional-encoding",title:"Sinusoidal Positional Encoding",section:"transformers",difficulty:"medium",description:`
## Sinusoidal Positional Encoding

Implement the positional encoding from the original Transformer paper.

### Formulas
\`\`\`
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Purpose
- Injects position information into embeddings
- Allows model to learn relative positions
- Fixed (not learned) encodings
    `,examples:[{input:"max_len=100, d_model=512",output:"PE matrix (100, 512)",explanation:"Position encoding for 100 positions"}],starterCode:`import numpy as np

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix (max_len, d_model)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape",input:"(50, 64)",expected:"(50, 64)",hidden:!1},{id:"2",description:"First position values",input:"(10, 4)",expected:"True",hidden:!0}],hints:["Create position indices: np.arange(max_len)","Create dimension indices for the divisor","Use div_term = 10000^(2i/d_model)","Apply sin to even indices, cos to odd indices"],solution:`import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
`},{id:"layer-norm",title:"Layer Normalization",section:"transformers",difficulty:"easy",description:`
## Layer Normalization

Implement layer normalization, used in Transformers instead of batch normalization.

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
\`\`\`

### Difference from BatchNorm
- **BatchNorm**: Normalizes across batch dimension
- **LayerNorm**: Normalizes across feature dimension
- LayerNorm works better for sequence models
    `,examples:[{input:"X (batch, seq_len, features)",output:"Normalized X, each position normalized independently",explanation:"Normalize across last dimension"}],starterCode:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.ones(4), np.zeros(4))",expected:"True",hidden:!1},{id:"2",description:"Gamma and beta work",input:"(np.array([[0, 0, 2, 2]]), np.array([1, 2, 1, 2]), np.zeros(4))",expected:"[[-1.0, -2.0, 1.0, 2.0]]",hidden:!0}],hints:["Compute mean and variance along the last axis","Use keepdims=True for proper broadcasting","Apply normalization: (x - mean) / sqrt(var + eps)"],solution:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    # Compute statistics along last dimension
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta
`},{id:"causal-mask",title:"Causal Attention Mask",section:"transformers",difficulty:"easy",description:`
## Causal (Autoregressive) Mask

Create a causal mask for decoder self-attention to prevent attending to future tokens.

### Mask Structure
\`\`\`
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
\`\`\`

### Usage
- Add mask to attention scores before softmax
- -inf becomes 0 after softmax
- Ensures each position only attends to previous positions
    `,examples:[{input:"seq_len=4",output:"Lower triangular mask (4, 4)",explanation:"Position i can only attend to positions <= i"}],starterCode:`import numpy as np

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
               0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape and pattern",input:"4",expected:"True",hidden:!1},{id:"2",description:"Works with attention",input:"3",expected:"[[0.0, -inf, -inf], [0.0, 0.0, -inf], [0.0, 0.0, 0.0]]",hidden:!0}],hints:["Create a matrix of ones using np.ones((seq_len, seq_len))","Use np.triu to get upper triangular matrix (excluding diagonal)","Replace 1s with -inf"],solution:`import numpy as np

def create_causal_mask(seq_len):
    # Create upper triangular matrix (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf
    mask = np.where(mask == 1, float('-inf'), 0.0)
    return mask
`}],uk=[{id:"vae-reparameterization",title:"VAE Reparameterization Trick",section:"generative-models",difficulty:"medium",description:`
## VAE Reparameterization Trick

Implement the reparameterization trick that allows backpropagation through stochastic sampling in VAEs.

### The Problem
We want to sample z ~ N(μ, σ²), but sampling is not differentiable.

### The Solution
\`\`\`
ε ~ N(0, 1)
z = μ + σ * ε
\`\`\`

### Why It Works
- ε is sampled independently of parameters
- z is now a deterministic function of μ, σ, and ε
- Gradients can flow through μ and σ
    `,examples:[{input:"mu = [0, 1], log_var = [0, 0]",output:"z = mu + exp(0.5 * log_var) * epsilon",explanation:"log_var=0 means σ=1"}],starterCode:`import numpy as np

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"(np.zeros((2, 4)), np.zeros((2, 4)))",expected:"(2, 4)",hidden:!1},{id:"2",description:"Zero variance returns mu",input:"zero_var_test",expected:"True",hidden:!0}],hints:["std = exp(0.5 * log_var)","Sample epsilon from standard normal","z = mu + std * epsilon"],solution:`import numpy as np

def reparameterize(mu, log_var):
    np.random.seed(42)
    # Compute standard deviation
    std = np.exp(0.5 * log_var)
    # Sample epsilon from standard normal
    eps = np.random.randn(*mu.shape)
    # Reparameterization trick
    z = mu + std * eps
    return z
`},{id:"vae-loss",title:"VAE Loss Function",section:"generative-models",difficulty:"medium",description:`
## VAE Loss (ELBO)

Implement the VAE loss function: reconstruction loss + KL divergence.

### Loss Components
\`\`\`
L = L_reconstruction + L_KL

L_reconstruction = MSE(x, x_reconstructed)
L_KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))
\`\`\`

### Intuition
- **Reconstruction loss**: Output should match input
- **KL divergence**: Latent distribution should be close to N(0,1)
    `,examples:[{input:"Perfect reconstruction, mu=0, var=1",output:"Loss ≈ 0",explanation:"Both terms are minimized"}],starterCode:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect case",input:"(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 2)), np.zeros((2, 2)))",expected:"(0.0, 0.0, 0.0)",hidden:!1},{id:"2",description:"Non-zero KL",input:"(np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 2)), np.zeros((1, 2)))",expected:"True",hidden:!0}],hints:["Reconstruction: np.mean((x - x_reconstructed)²)","KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))","Average over batch"],solution:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = np.mean((x - x_reconstructed) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    total_loss = recon_loss + kl_loss

    return round(total_loss, 4), round(recon_loss, 4), round(kl_loss, 4)
`},{id:"diffusion-noise-schedule",title:"Diffusion Noise Schedule",section:"generative-models",difficulty:"easy",description:`
## Diffusion Noise Schedule

Implement a linear noise schedule for diffusion models.

### Linear Schedule
\`\`\`
β_t = β_start + t * (β_end - β_start) / T

where t = 0, 1, ..., T-1
\`\`\`

### Derived Quantities
\`\`\`
α_t = 1 - β_t
ᾱ_t = prod(α_1, ..., α_t)  # cumulative product
\`\`\`

These control how much noise is added at each step.
    `,examples:[{input:"T=1000, beta_start=0.0001, beta_end=0.02",output:"betas, alphas, alpha_bars arrays",explanation:"Standard DDPM schedule"}],starterCode:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Beta range correct",input:"(100, 0.0001, 0.02)",expected:"True",hidden:!1},{id:"2",description:"Alpha bar decreases",input:"(50, 0.001, 0.01)",expected:"True",hidden:!0}],hints:["Use np.linspace for linear interpolation","alphas = 1 - betas","alpha_bars = np.cumprod(alphas)"],solution:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    # Linear schedule
    betas = np.linspace(beta_start, beta_end, T)

    # Compute alphas
    alphas = 1 - betas

    # Cumulative product
    alpha_bars = np.cumprod(alphas)

    return betas, alphas, alpha_bars
`},{id:"diffusion-forward",title:"Diffusion Forward Process",section:"generative-models",difficulty:"medium",description:`
## Diffusion Forward Process (Adding Noise)

Implement the forward diffusion process that adds noise to data.

### Formula
\`\`\`
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

where ε ~ N(0, I)
\`\`\`

### Intuition
- As t increases, ᾱ_t decreases
- More noise is added, signal is reduced
- At t=T, x_T ≈ pure noise
    `,examples:[{input:"x_0 (image), t=500, T=1000",output:"x_t (noisy image)",explanation:"Halfway through diffusion process"}],starterCode:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    np.random.seed(42)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"(np.random.randn(2, 4), 50, np.linspace(0.99, 0.01, 100))",expected:"(2, 4)",hidden:!1},{id:"2",description:"t=0 returns near original",input:"t_zero_test",expected:"True",hidden:!0}],hints:["Get alpha_bar_t = alpha_bars[t]","Sample noise from standard normal","Apply formula: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"],solution:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    np.random.seed(42)

    alpha_bar_t = alpha_bars[t]

    # Sample noise
    noise = np.random.randn(*x_0.shape)

    # Forward process
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
`},{id:"vqvae-quantization",title:"VQ-VAE Vector Quantization",section:"generative-models",difficulty:"hard",description:`
## VQ-VAE Vector Quantization

Implement the vector quantization layer used in VQ-VAE (Vector Quantized Variational Autoencoder).

### Overview
Unlike standard VAEs that use continuous latent spaces, VQ-VAE uses **discrete** latent representations by mapping encoder outputs to the nearest embedding in a learned codebook.

### Quantization Process
1. **Encoder output**: z_e of shape (batch, H, W, D)
2. **Codebook**: K embedding vectors of dimension D
3. **Find nearest**: For each spatial position, find the closest codebook entry
4. **Quantize**: Replace z_e with the nearest codebook vector z_q

### Distance Calculation
\`\`\`
distances[b, h, w, k] = ||z_e[b, h, w] - codebook[k]||²
\`\`\`

### Straight-Through Estimator
During training, gradients flow through z_q to z_e by copying gradients:
\`\`\`
z_q = z_e + stop_gradient(z_q - z_e)
\`\`\`
(We don't implement this here, just the forward pass)

### VQ-VAE Loss Components
\`\`\`
L = reconstruction + β * ||sg[z_e] - e||² + ||z_e - sg[e]||²
\`\`\`
- **Codebook loss**: Moves embeddings toward encoder outputs
- **Commitment loss**: Commits encoder to embeddings

### Function Signature
\`\`\`python
def vq_quantize(z_e, codebook):
    # z_e: (batch, H, W, embedding_dim) - encoder output
    # codebook: (num_embeddings, embedding_dim) - K embedding vectors
    # Returns: z_q, indices
\`\`\`
    `,examples:[{input:"z_e (2, 4, 4, 64), codebook (512, 64)",output:"z_q (2, 4, 4, 64), indices (2, 4, 4)",explanation:"Each spatial position mapped to one of 512 codes"}],starterCode:`import numpy as np

def vq_quantize(z_e, codebook):
    """
    Vector quantization for VQ-VAE.

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        codebook: Embedding vectors (num_embeddings, embedding_dim)

    Returns:
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        indices: Codebook indices used (batch, H, W)
    """
    # Your code here
    pass


def vq_loss(z_e, z_q, beta=0.25):
    """
    Compute VQ-VAE losses (codebook + commitment).

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        beta: Commitment loss weight (default: 0.25)

    Returns:
        codebook_loss: Loss to update codebook
        commitment_loss: Loss to commit encoder to codebook
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))",expected:"((2, 4, 4, 8), (2, 4, 4))",hidden:!1},{id:"2",description:"Indices are valid codebook indices",input:"valid_indices_test",expected:"True",hidden:!1},{id:"3",description:"Quantized vectors come from codebook",input:"vectors_from_codebook_test",expected:"True",hidden:!0},{id:"4",description:"VQ loss computation correct",input:"vq_loss_test",expected:"True",hidden:!0}],hints:["Reshape z_e to (batch*H*W, D) for easier distance computation","Use broadcasting: ||a - b||² = ||a||² + ||b||² - 2*a·b","np.argmin along the codebook axis gives indices","Index into codebook with the indices to get z_q","Reshape back to original spatial dimensions","For loss: codebook_loss = ||z_e.detach() - z_q||², commitment = ||z_e - z_q.detach()||²"],solution:`import numpy as np

def vq_quantize(z_e, codebook):
    z_e = np.array(z_e)
    codebook = np.array(codebook)

    batch, H, W, D = z_e.shape
    K, _ = codebook.shape

    # Flatten spatial dimensions: (batch*H*W, D)
    z_flat = z_e.reshape(-1, D)

    # Compute distances using: ||a-b||² = ||a||² + ||b||² - 2*a·b
    # z_flat: (N, D), codebook: (K, D)
    z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
    codebook_sq = np.sum(codebook ** 2, axis=1)        # (K,)
    cross = z_flat @ codebook.T                        # (N, K)

    distances = z_sq + codebook_sq - 2 * cross         # (N, K)

    # Find nearest codebook entry
    indices_flat = np.argmin(distances, axis=1)        # (N,)

    # Get quantized vectors
    z_q_flat = codebook[indices_flat]                  # (N, D)

    # Reshape back
    z_q = z_q_flat.reshape(batch, H, W, D)
    indices = indices_flat.reshape(batch, H, W)

    return z_q, indices


def vq_loss(z_e, z_q, beta=0.25):
    z_e = np.array(z_e)
    z_q = np.array(z_q)

    # Codebook loss: ||sg[z_e] - z_q||² (moves codebook toward encoder output)
    # In practice, sg[z_e] means z_e is treated as constant
    codebook_loss = np.mean((z_e - z_q) ** 2)

    # Commitment loss: ||z_e - sg[z_q]||² (commits encoder to codebook)
    # Same computation, but gradient only flows to z_e
    commitment_loss = beta * np.mean((z_e - z_q) ** 2)

    return round(codebook_loss, 4), round(commitment_loss, 4)
`},{id:"kl-divergence",title:"KL Divergence (Gaussians)",section:"generative-models",difficulty:"easy",description:`
## KL Divergence Between Gaussians

Compute the KL divergence between two univariate Gaussian distributions.

### Formula
\`\`\`
KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
\`\`\`

Where P = N(μ_p, σ_p²) and Q = N(μ_q, σ_q²)

### Properties
- KL ≥ 0
- KL = 0 iff P = Q
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
    `,examples:[{input:"P = N(0, 1), Q = N(0, 1)",output:"0",explanation:"Same distribution"}],starterCode:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P
        mu_q, sigma_q: Mean and std of Q

    Returns:
        kl: KL(P || Q)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same distribution",input:"(0, 1, 0, 1)",expected:"0.0",hidden:!1},{id:"2",description:"Different means",input:"(1, 1, 0, 1)",expected:"0.5",hidden:!1},{id:"3",description:"Different variances",input:"(0, 2, 0, 1)",expected:"0.4431",hidden:!0}],hints:["Apply the formula directly","Use np.log for natural logarithm","Remember: σ² is variance, σ is std"],solution:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5

    kl = term1 + term2 + term3
    return round(kl, 4)
`}],ck=[{id:"numpy-array-creation",title:"Array Creation Methods",section:"numpy-fundamentals",difficulty:"easy",description:`
## NumPy Array Creation

Implement a function that creates various NumPy arrays using different methods.

### Requirements
Create and return a dictionary with:
- \`zeros\`: 3x4 array of zeros
- \`ones\`: 2x3 array of ones
- \`arange\`: array from 0 to 9
- \`linspace\`: 5 evenly spaced values from 0 to 1
- \`eye\`: 4x4 identity matrix

### Function Signature
\`\`\`python
def create_arrays() -> dict:
\`\`\`
    `,examples:[{input:"create_arrays()",output:"{'zeros': array([[0,0,0,0],...]), ...}",explanation:"Dictionary with 5 different array types"}],starterCode:`import numpy as np

def create_arrays() -> dict:
    """
    Create various NumPy arrays.

    Returns:
        Dictionary with keys: zeros, ones, arange, linspace, eye
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape correct",input:"zeros_shape",expected:"(3, 4)",hidden:!1},{id:"2",description:"Linspace length correct",input:"linspace_len",expected:"5",hidden:!1},{id:"3",description:"Eye is identity",input:"eye_check",expected:"True",hidden:!0}],hints:["Use np.zeros((3, 4)) for zero array","np.linspace(start, stop, num) creates evenly spaced values","np.eye(n) creates an identity matrix"],solution:`import numpy as np

def create_arrays() -> dict:
    return {
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4)
    }
`},{id:"numpy-indexing",title:"Advanced Indexing",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Advanced Indexing

Implement various indexing operations on a 2D array.

### Given
\`\`\`python
arr = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])
\`\`\`

### Requirements
Return a dictionary with:
- \`row_1\`: Second row (index 1)
- \`col_2\`: Third column (index 2)
- \`subarray\`: Rows 1-2, columns 2-4
- \`diagonal\`: Main diagonal
- \`reversed\`: Array with rows reversed

### Function Signature
\`\`\`python
def advanced_indexing(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr = np.arange(20).reshape(4, 5)",output:"{'row_1': [5,6,7,8,9], ...}",explanation:"Various slicing and indexing operations"}],starterCode:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    """
    Perform advanced indexing operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with indexed results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row extraction",input:"(np.arange(20).reshape(4, 5),)",expected:"row_check",hidden:!1},{id:"2",description:"Subarray correct",input:"subarray_check",expected:"True",hidden:!1}],hints:["arr[1] or arr[1, :] gets row 1","arr[:, 2] gets column 2","arr[1:3, 2:5] gets a subarray","np.diag(arr) extracts diagonal","arr[::-1] reverses rows"],solution:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    return {
        'row_1': arr[1],
        'col_2': arr[:, 2],
        'subarray': arr[1:3, 2:5],
        'diagonal': np.diag(arr),
        'reversed': arr[::-1]
    }
`},{id:"numpy-broadcasting",title:"Broadcasting Operations",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Broadcasting

Implement operations that leverage NumPy's broadcasting rules.

### Task
Given a 2D array of shape (3, 4):
1. Subtract the row means from each row (center rows)
2. Subtract the column means from each column (center columns)
3. Add a 1D bias vector of shape (4,) to each row
4. Multiply by a column vector of shape (3, 1)

### Broadcasting Rules
- Arrays are compatible when dimensions are equal or one of them is 1
- Operations proceed element-wise on matching dimensions

### Function Signature
\`\`\`python
def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr (3,4), bias (4,), scale (3,1)",output:"{'row_centered': ..., 'col_centered': ..., 'biased': ..., 'scaled': ...}",explanation:"Various broadcasting operations"}],starterCode:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    """
    Perform broadcasting operations.

    Args:
        arr: 2D array of shape (3, 4)
        bias: 1D array of shape (4,)
        scale: 2D array of shape (3, 1)

    Returns:
        Dictionary with results of broadcasting operations
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row centering",input:"row_center_test",expected:"True",hidden:!1},{id:"2",description:"Bias addition",input:"bias_test",expected:"True",hidden:!1}],hints:["Row means: arr.mean(axis=1, keepdims=True)","Column means: arr.mean(axis=0, keepdims=True) or arr.mean(axis=0)","Bias shape (4,) broadcasts with arr shape (3, 4)","Scale shape (3, 1) broadcasts with arr shape (3, 4)"],solution:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    # Row centered: subtract mean of each row
    row_means = arr.mean(axis=1, keepdims=True)  # (3, 1)
    row_centered = arr - row_means

    # Column centered: subtract mean of each column
    col_means = arr.mean(axis=0)  # (4,)
    col_centered = arr - col_means

    # Add bias to each row
    biased = arr + bias  # bias (4,) broadcasts to (3, 4)

    # Scale each row
    scaled = arr * scale  # scale (3, 1) broadcasts to (3, 4)

    return {
        'row_centered': row_centered,
        'col_centered': col_centered,
        'biased': biased,
        'scaled': scaled
    }
`},{id:"numpy-aggregations",title:"Aggregation Functions",section:"numpy-fundamentals",difficulty:"easy",description:`
## NumPy Aggregations

Implement common aggregation operations along different axes.

### Task
Given a 2D array, compute:
- Global statistics (mean, std, min, max, sum)
- Row-wise statistics (along axis=1)
- Column-wise statistics (along axis=0)
- Argmax and argmin (indices of max/min values)

### Function Signature
\`\`\`python
def compute_aggregations(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"np.array([[1, 2, 3], [4, 5, 6]])",output:"{'global_mean': 3.5, 'row_sum': [6, 15], ...}",explanation:"Various aggregation results"}],starterCode:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    """
    Compute aggregation statistics.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with aggregation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Global mean",input:"(np.array([[1, 2, 3], [4, 5, 6]]),)",expected:"mean_check",hidden:!1},{id:"2",description:"Row sums",input:"row_sum_check",expected:"True",hidden:!1}],hints:["arr.mean() computes global mean","arr.mean(axis=0) computes column means","arr.mean(axis=1) computes row means","np.argmax(arr) gives index of max in flattened array"],solution:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    return {
        'global_mean': arr.mean(),
        'global_std': arr.std(),
        'global_min': arr.min(),
        'global_max': arr.max(),
        'global_sum': arr.sum(),
        'row_mean': arr.mean(axis=1),
        'row_sum': arr.sum(axis=1),
        'col_mean': arr.mean(axis=0),
        'col_sum': arr.sum(axis=0),
        'argmax': np.unravel_index(arr.argmax(), arr.shape),
        'argmin': np.unravel_index(arr.argmin(), arr.shape),
    }
`},{id:"numpy-reshape-transpose",title:"Reshape and Transpose",section:"numpy-fundamentals",difficulty:"medium",description:`
## Reshape and Transpose Operations

Master array shape manipulation - essential for ML data pipelines.

### Task
Given a 1D array of 24 elements:
1. Reshape to (4, 6)
2. Reshape to (2, 3, 4)
3. Transpose the 2D version
4. Swap axes on the 3D version
5. Flatten back to 1D

### Important Concepts
- \`reshape\` changes shape without changing data order
- \`transpose\` swaps axes
- \`-1\` in reshape means "infer this dimension"

### Function Signature
\`\`\`python
def reshape_transpose(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"np.arange(24)",output:"{'shape_4x6': (4,6), 'shape_2x3x4': (2,3,4), ...}",explanation:"Various reshaping results"}],starterCode:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    """
    Perform reshape and transpose operations.

    Args:
        arr: 1D array of 24 elements

    Returns:
        Dictionary with reshaped arrays
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2D reshape shape",input:"(np.arange(24),)",expected:"shape_2d_check",hidden:!1},{id:"2",description:"3D reshape shape",input:"shape_3d_check",expected:"True",hidden:!1}],hints:["arr.reshape(4, 6) or arr.reshape(4, -1)","arr.reshape(2, 3, 4) for 3D",".T or .transpose() for transpose",".flatten() or .ravel() for 1D"],solution:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    # 2D reshape
    arr_2d = arr.reshape(4, 6)

    # 3D reshape
    arr_3d = arr.reshape(2, 3, 4)

    # Transpose 2D
    arr_2d_T = arr_2d.T

    # Swap axes on 3D (swap axis 0 and 2)
    arr_3d_swapped = arr_3d.transpose(2, 1, 0)

    # Flatten
    arr_flat = arr_2d.flatten()

    return {
        'arr_2d': arr_2d,
        'arr_3d': arr_3d,
        'arr_2d_transposed': arr_2d_T,
        'arr_3d_swapped': arr_3d_swapped,
        'arr_flat': arr_flat
    }
`}],dk=[{id:"einsum-basics",title:"Einsum Fundamentals",section:"einsum",difficulty:"easy",description:`
## Einsum Fundamentals

Einstein summation notation provides a powerful way to express array operations.

### Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

### Basic Operations
- \`'i->'\`: Sum all elements (reduce)
- \`'ij->i'\`: Sum along columns (row sums)
- \`'ij->j'\`: Sum along rows (column sums)
- \`'ii->i'\`: Extract diagonal
- \`'ij->ji'\`: Transpose

### Task
Implement basic einsum operations on a 2D array.
    `,examples:[{input:"np.array([[1,2],[3,4]])",output:"{'sum': 10, 'row_sum': [3,7], 'transpose': [[1,3],[2,4]]}",explanation:"Basic einsum operations"}],starterCode:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Sum all elements",input:"(np.array([[1, 2], [3, 4]]),)",expected:"sum_check",hidden:!1},{id:"2",description:"Transpose",input:"transpose_check",expected:"True",hidden:!1}],hints:["'ij->' sums all elements","'ij->i' sums each row","'ij->ji' transposes","'ii->i' extracts diagonal (for square matrices)"],solution:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    return {
        'sum_all': np.einsum('ij->', arr),
        'row_sum': np.einsum('ij->i', arr),
        'col_sum': np.einsum('ij->j', arr),
        'transpose': np.einsum('ij->ji', arr),
        'diagonal': np.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None
    }
`},{id:"einsum-matrix-ops",title:"Matrix Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Matrix Operations with Einsum

Einsum can express matrix multiplication and more complex operations.

### Key Operations
- \`'ik,kj->ij'\`: Matrix multiplication (A @ B)
- \`'ij,ij->ij'\`: Element-wise multiplication (Hadamard)
- \`'ij,ij->'\`: Frobenius inner product (sum of element-wise product)
- \`'ij,j->i'\`: Matrix-vector multiplication
- \`'i,j->ij'\`: Outer product

### Task
Implement matrix operations using einsum.

### Function Signature
\`\`\`python
def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"A (2,3), B (3,2), v (3,)",output:"{'matmul': A@B, 'matvec': A@v, ...}",explanation:"Matrix operations via einsum"}],starterCode:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Matrix multiplication",input:"(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))",expected:"matmul_check",hidden:!1},{id:"2",description:"Outer product",input:"outer_check",expected:"True",hidden:!1}],hints:["'ik,kj->ij' contracts over k (matrix multiply)","'ij,j->i' contracts vector with columns","'i,j->ij' creates outer product (no contraction)"],solution:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    return {
        'matmul': np.einsum('ik,kj->ij', A, B),
        'matvec': np.einsum('ij,j->i', A, v),
        'outer_product': np.einsum('i,j->ij', v, v),
        'hadamard': np.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': np.einsum('ij,ij->', A, A),   # sum(A * A)
        'trace': np.einsum('ii->', A) if A.shape[0] == A.shape[1] else None
    }
`},{id:"einsum-batch-ops",title:"Batch Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Batch Operations with Einsum

Einsum shines for batch operations - multiple matrices at once.

### Batch Matrix Operations
- \`'bij,bjk->bik'\`: Batch matrix multiplication
- \`'bij,bj->bi'\`: Batch matrix-vector multiplication
- \`'bij->bji'\`: Batch transpose

### Attention-style Operations
- \`'bqd,bkd->bqk'\`: Query-Key dot products (attention scores)
- \`'bqk,bkv->bqv'\`: Weighted sum of values

### Task
Implement batch operations common in deep learning.
    `,examples:[{input:"batch of 4 matrices (4, 3, 3)",output:"Batch-wise operations",explanation:"Operations applied independently to each batch"}],starterCode:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Attention scores shape",input:"(np.random.randn(2, 4, 8), np.random.randn(2, 6, 8), np.random.randn(2, 6, 16))",expected:"scores_shape_check",hidden:!1},{id:"2",description:"Output shape",input:"output_shape_check",expected:"True",hidden:!1}],hints:["'bqd,bkd->bqk' computes Q @ K.T for each batch","'bqk,bkv->bqv' applies attention weights to values","b is the batch dimension, preserved in output"],solution:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    # Q: (batch, seq_q, dim)
    # K: (batch, seq_k, dim)
    # V: (batch, seq_k, dim_v)

    # Attention scores: Q @ K.T for each batch
    # Result: (batch, seq_q, seq_k)
    scores = np.einsum('bqd,bkd->bqk', Q, K)

    # Scale scores
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax (simplified, along last axis)
    exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # Result: (batch, seq_q, dim_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }
`},{id:"einsum-advanced",title:"Advanced Einsum Patterns",section:"einsum",difficulty:"hard",description:`
## Advanced Einsum Patterns

Complex tensor operations for deep learning.

### Multi-Head Attention Pattern
\`\`\`python
# Split heads: (batch, seq, heads, dim)
# Attention per head: 'bhqd,bhkd->bhqk'
# Combine heads: 'bhqv->bqhv' then reshape
\`\`\`

### Bilinear Operations
\`\`\`python
# x.T @ W @ y: 'i,ijk,k->j'
# Batch bilinear: 'bi,ijk,bk->bj'
\`\`\`

### Task
Implement multi-head attention using einsum.
    `,examples:[{input:"Q, K, V with multiple heads",output:"Multi-head attention output",explanation:"Parallel attention across heads"}],starterCode:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"(np.random.randn(2, 4, 64), np.random.randn(2, 4, 64), np.random.randn(2, 4, 64), 8)",expected:"(2, 4, 64)",hidden:!1},{id:"2",description:"Attention computed correctly",input:"attention_check",expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (batch, seq, num_heads, d_k)","Use 'bqhd,bkhd->bhqk' for attention scores per head","Use 'bhqk,bkhd->bqhd' for weighted values","Reshape back to (batch, seq, d_model)"],solution:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Reshape to (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Compute attention scores for all heads
    # (batch, seq_q, heads, d_k) x (batch, seq_k, heads, d_k) -> (batch, heads, seq_q, seq_k)
    scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # (batch, heads, seq_q, seq_k) x (batch, seq_k, heads, d_k) -> (batch, seq_q, heads, d_k)
    output = np.einsum('bhqk,bkhd->bqhd', attention, V)

    # Reshape back to (batch, seq_q, d_model)
    output = output.reshape(batch, seq_q, d_model)

    return output
`},{id:"einsum-vs-matmul",title:"Einsum vs Traditional Operations",section:"einsum",difficulty:"easy",description:"\n## Einsum vs Traditional Operations\n\nCompare einsum with equivalent NumPy operations.\n\n### Equivalences\n| Einsum | NumPy |\n|--------|-------|\n| `'ij->'` | `np.sum(A)` |\n| `'ij->i'` | `np.sum(A, axis=1)` |\n| `'ik,kj->ij'` | `A @ B` |\n| `'ij,ij->'` | `np.sum(A * B)` |\n| `'ij->ji'` | `A.T` |\n\n### Task\nVerify einsum produces same results as traditional operations.\n    ",examples:[{input:"Two matrices A, B",output:"Comparison showing equivalence",explanation:"All operations produce identical results"}],starterCode:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare einsum with traditional NumPy operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' results for comparison
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"All results match",input:"(np.random.randn(3, 4), np.random.randn(4, 5))",expected:"True",hidden:!1}],hints:["Use np.allclose() to compare floating point arrays","Remember 'ik,kj->ij' is matrix multiplication"],solution:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    results = {
        'sum': {
            'einsum': np.einsum('ij->', A),
            'numpy': np.sum(A)
        },
        'row_sum': {
            'einsum': np.einsum('ij->i', A),
            'numpy': np.sum(A, axis=1)
        },
        'matmul': {
            'einsum': np.einsum('ik,kj->ij', A, B),
            'numpy': A @ B
        },
        'transpose': {
            'einsum': np.einsum('ij->ji', A),
            'numpy': A.T
        },
        'hadamard_sum': {
            'einsum': np.einsum('ij,ij->', A, A),
            'numpy': np.sum(A * A)
        }
    }

    # Verify all match
    all_match = all(
        np.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`}],pk=[{id:"tensor-creation",title:"Tensor Creation (NumPy Style)",section:"pytorch-basics",difficulty:"easy",description:"\n## Tensor Creation\n\nLearn tensor creation patterns used in PyTorch, implemented with NumPy.\n\n### PyTorch Equivalents\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.tensor([1,2,3])` | `np.array([1,2,3])` |\n| `torch.zeros(3, 4)` | `np.zeros((3, 4))` |\n| `torch.randn(2, 3)` | `np.random.randn(2, 3)` |\n| `torch.arange(10)` | `np.arange(10)` |\n| `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` |\n\n### Data Types\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.float32` | `np.float32` |\n| `torch.int64` | `np.int64` |\n\n### Task\nCreate tensors matching PyTorch patterns.\n    ",examples:[{input:"create_tensors()",output:"Dictionary of tensors",explanation:"Various tensor creation methods"}],starterCode:`import numpy as np

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    np.random.seed(42)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape",input:"zeros_check",expected:"True",hidden:!1},{id:"2",description:"Random tensor shape",input:"randn_check",expected:"True",hidden:!1}],hints:["np.zeros((3, 4)) for 3x4 zeros","np.random.randn(2, 3) for random normal","Specify dtype with astype() or dtype parameter"],solution:`import numpy as np

def create_tensors() -> dict:
    np.random.seed(42)

    return {
        'from_list': np.array([1, 2, 3, 4], dtype=np.float32),
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'randn': np.random.randn(2, 3),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4),
        'full': np.full((2, 3), 7.0),
    }
`},{id:"tensor-operations",title:"Tensor Operations",section:"pytorch-basics",difficulty:"easy",description:`
## Basic Tensor Operations

Common tensor operations used in neural networks.

### Arithmetic Operations
- Element-wise: +, -, *, /
- Matrix multiplication: @ or np.matmul
- Power: ** or np.power

### Reduction Operations
- sum, mean, std, var
- min, max, argmin, argmax

### Shape Operations
- reshape, view (reshape in NumPy)
- squeeze, unsqueeze (np.squeeze, np.expand_dims)
- permute, transpose

### Task
Implement common tensor operations.
    `,examples:[{input:"tensor_ops(x, y)",output:"Dictionary with operation results",explanation:"Various tensor operations"}],starterCode:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First tensor (2, 3)
        y: Second tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Addition",input:"(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))",expected:"add_check",hidden:!1},{id:"2",description:"Mean computation",input:"mean_check",expected:"True",hidden:!1}],hints:["Element-wise ops work directly: x + y, x * y","Use axis parameter for reductions along specific dims","np.expand_dims(x, axis=0) adds dimension"],solution:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(axis=0),
        'sum_axis1': x.sum(axis=1),
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': np.expand_dims(x, axis=0),  # (1, 2, 3)
        'squeeze': np.squeeze(np.expand_dims(x, 0)),
        'transpose': x.T,
    }
`},{id:"autograd-concepts",title:"Autograd Concepts (Manual)",section:"pytorch-basics",difficulty:"medium",description:`
## Autograd Concepts

Understand automatic differentiation by implementing it manually.

### PyTorch Autograd
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
\`\`\`

### Manual Gradient Computation
For y = x² + 3x + 1:
- dy/dx = 2x + 3

### Task
Implement forward pass and gradient computation for simple functions.
    `,examples:[{input:"compute_gradients(x=2.0)",output:"{'y': 11.0, 'grad': 7.0}",explanation:"Forward and backward pass"}],starterCode:`import numpy as np

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx)
    """
    # Your code here
    pass


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients for X, W, and b
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple gradient",input:"(2.0,)",expected:"grad_check",hidden:!1},{id:"2",description:"Linear layer gradients",input:"linear_grad_check",expected:"True",hidden:!1}],hints:["For y = x² + 3x + 1, dy/dx = 2x + 3","For Y = X @ W + b: dL/dW = X.T @ grad_output","dL/dX = grad_output @ W.T","dL/db = sum(grad_output, axis=0)"],solution:`import numpy as np

def compute_gradients(x: float) -> dict:
    # Forward pass
    y = x ** 2 + 3 * x + 1

    # Backward pass (analytical gradient)
    grad = 2 * x + 3

    return {'y': y, 'grad': grad}


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    # Forward: Y = X @ W + b

    # Backward:
    # dL/dW = X.T @ grad_output
    grad_W = X.T @ grad_output

    # dL/db = sum of grad_output over batch
    grad_b = grad_output.sum(axis=0)

    # dL/dX = grad_output @ W.T
    grad_X = grad_output @ W.T

    return {
        'grad_W': grad_W,
        'grad_b': grad_b,
        'grad_X': grad_X
    }
`},{id:"nn-modules",title:"Neural Network Modules",section:"pytorch-basics",difficulty:"medium",description:`
## Neural Network Modules

Implement PyTorch-style nn.Module patterns in NumPy.

### Module Pattern
\`\`\`python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
\`\`\`

### Task
Implement Linear and ReLU modules with forward method.
    `,examples:[{input:"Linear(10, 5).forward(x)",output:"Output of shape (batch, 5)",explanation:"Linear transformation"}],starterCode:`import numpy as np

class Linear:
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Initialize weights with small random values
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass

    def parameters(self):
        # Return list of parameters
        pass


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


class Sequential:
    """Container for sequential layers"""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Linear output shape",input:"linear_shape_check",expected:"True",hidden:!1},{id:"2",description:"Sequential forward",input:"sequential_check",expected:"True",hidden:!1}],hints:["Initialize W with np.random.randn() * 0.01","ReLU: np.maximum(0, x)","Sequential applies layers in order"],solution:`import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim)
    )
`},{id:"loss-functions",title:"Loss Functions",section:"pytorch-basics",difficulty:"medium",description:`
## Common Loss Functions

Implement loss functions used in deep learning.

### Cross-Entropy Loss
\`\`\`
L = -sum(y * log(softmax(logits)))
\`\`\`

### MSE Loss
\`\`\`
L = mean((y_pred - y_true)²)
\`\`\`

### Binary Cross-Entropy
\`\`\`
L = -mean(y * log(p) + (1-y) * log(1-p))
\`\`\`

### Task
Implement these loss functions with numerical stability.
    `,examples:[{input:"cross_entropy(logits, targets)",output:"Scalar loss value",explanation:"Cross-entropy for classification"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Your code here
    pass


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy with numerical stability."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Cross-entropy computation",input:"(np.array([[2.0, 1.0, 0.1]]), np.array([0]))",expected:"ce_check",hidden:!1},{id:"2",description:"MSE computation",input:"mse_check",expected:"True",hidden:!1}],hints:["Softmax: exp(x - max(x)) / sum(exp(x - max(x)))","Use np.clip for numerical stability in log","Index logits with targets for cross-entropy"],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]

    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    return loss


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
`}],fk=[{id:"e2e-mlp",title:"E2E: 2-Layer MLP with Backprop",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End 2-Layer MLP

Implement a complete 2-layer MLP with forward pass, backward pass, and training loop.

### Architecture
\`\`\`
Input (batch, input_dim)
  ↓
Linear(input_dim, hidden_dim) + ReLU
  ↓
Linear(hidden_dim, output_dim)
  ↓
Softmax + Cross-Entropy Loss
\`\`\`

### Forward Pass
\`\`\`python
h = relu(X @ W1 + b1)
logits = h @ W2 + b2
loss = cross_entropy(softmax(logits), y)
\`\`\`

### Backward Pass (Chain Rule)
\`\`\`python
# Output layer
d_logits = softmax(logits) - y_onehot  # (batch, output_dim)
d_W2 = h.T @ d_logits
d_b2 = d_logits.sum(axis=0)

# Hidden layer
d_h = d_logits @ W2.T
d_relu = d_h * (h > 0)  # ReLU derivative
d_W1 = X.T @ d_relu
d_b1 = d_relu.sum(axis=0)
\`\`\`

### Task
Implement forward, backward, and training step.
    `,examples:[{input:"MLP(784, 128, 10) on MNIST-like data",output:"Trained model with decreasing loss",explanation:"Full training loop"}],starterCode:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize 2-layer MLP with random weights."""
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Cache for backward pass
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            X: Input (batch, input_dim)

        Returns:
            logits: Output (batch, output_dim)
        """
        # Your code here
        pass

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch, output_dim)
            y: True labels (batch,) as integers

        Returns:
            loss: Scalar loss value
        """
        # Your code here
        pass

    def backward(self, y: np.ndarray) -> dict:
        """
        Backward pass.

        Args:
            y: True labels (batch,) as integers

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        # Your code here
        pass

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step: forward, loss, backward, update.

        Returns:
            loss: Loss value before update
        """
        # Your code here
        pass


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    """Train MLP and return loss history."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward pass shape",input:"forward_shape_check",expected:"True",hidden:!1},{id:"2",description:"Loss decreases",input:"loss_decrease_check",expected:"True",hidden:!1},{id:"3",description:"Gradients computed",input:"gradients_check",expected:"True",hidden:!0}],hints:["Cache X, h (hidden activations), and logits in forward pass","d_logits = probs - y_onehot (softmax gradient combined with cross-entropy)","ReLU gradient: 1 if x > 0, else 0","Update: W = W - lr * dW"],solution:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Layer 1: Linear + ReLU
        z1 = X @ self.W1 + self.b1
        h = self.relu(z1)

        # Layer 2: Linear
        logits = h @ self.W2 + self.b2

        # Cache for backward
        self.cache = {'X': X, 'z1': z1, 'h': h, 'logits': logits}

        return logits

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        probs = self.softmax(logits)
        batch_size = logits.shape[0]
        correct_probs = probs[np.arange(batch_size), y]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-15, 1)))
        self.cache['probs'] = probs
        return loss

    def backward(self, y: np.ndarray) -> dict:
        X = self.cache['X']
        h = self.cache['h']
        z1 = self.cache['z1']
        probs = self.cache['probs']
        batch_size = X.shape[0]

        # One-hot encode targets
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1

        # Output layer gradients
        d_logits = (probs - y_onehot) / batch_size
        dW2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)

        # Hidden layer gradients
        d_h = d_logits @ self.W2.T
        d_relu = d_h * (z1 > 0)  # ReLU derivative
        dW1 = X.T @ d_relu
        db1 = d_relu.sum(axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        logits = self.forward(X)
        loss = self.compute_loss(logits, y)
        grads = self.backward(y)

        # Update weights
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

        return loss


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = MLP(input_dim, 64, output_dim)

    losses = []
    for _ in range(epochs):
        loss = model.train_step(X_train, y_train, lr)
        losses.append(loss)

    return model, losses
`},{id:"e2e-transformer",title:"E2E: Transformer Encoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Transformer Encoder

Implement a complete Transformer encoder block.

### Architecture
\`\`\`
Input Embeddings + Positional Encoding
  ↓
Multi-Head Self-Attention
  ↓
Add & LayerNorm (Residual)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & LayerNorm (Residual)
  ↓
Output
\`\`\`

### Components
1. **Positional Encoding**: Sinusoidal position embeddings
2. **Multi-Head Attention**: Parallel attention heads
3. **FFN**: Two linear layers with ReLU
4. **Layer Normalization**: Normalize across features
5. **Residual Connections**: Add input to output

### Task
Implement all components and combine into encoder block.
    `,examples:[{input:"TransformerEncoder(d_model=64, nhead=8, dim_ff=256)",output:"Encoded sequence (batch, seq_len, d_model)",explanation:"Complete encoder forward pass"}],starterCode:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    # Your code here
    pass


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    # Your code here
    pass


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    # Your code here
    pass


def multi_head_attention(Q, K, V, num_heads):
    """Multi-head attention."""
    # Your code here
    pass


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear"""
    # Your code here
    pass


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through encoder block.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Your code here
        pass


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    """Stack multiple encoder blocks."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"shape_check",expected:"True",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"attention_check",expected:"True",hidden:!1}],hints:["Attention: softmax(QK^T / sqrt(d_k)) @ V","Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta","Residual: output = LayerNorm(x + Sublayer(x))","FFN: max(0, xW1 + b1)W2 + b2"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, axis=-1)
    return attention @ V, attention


def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    d_k = Q.shape[-1]
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
    attention = softmax(scores, axis=-1)
    output = np.einsum('bhqk,bhkd->bhqd', attention, V)

    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return output


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # FFN parameters
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-head self-attention + residual + layer norm
        attn_output = multi_head_attention(x, x, x, self.num_heads)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward + residual + layer norm
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)

        return x


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    batch, seq_len, _ = x.shape

    # Add positional encoding
    PE = positional_encoding(seq_len, d_model)
    x = x + PE

    # Stack encoder blocks
    for _ in range(num_layers):
        block = TransformerEncoderBlock(d_model, num_heads, d_ff)
        x = block.forward(x)

    return x
`},{id:"e2e-vae",title:"E2E: Variational Autoencoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Variational Autoencoder

Implement a complete VAE with encoder, decoder, reparameterization, and ELBO loss.

### Architecture
\`\`\`
Encoder: x → h → (μ, log_σ²)
         ↓
Reparameterization: z = μ + σ * ε, ε ~ N(0,1)
         ↓
Decoder: z → h → x_reconstructed
\`\`\`

### Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_hat||² + KL(q(z|x) || p(z))
KL = -0.5 * sum(1 + log_var - μ² - exp(log_var))
\`\`\`

### Task
Implement encoder, decoder, reparameterization trick, and loss computation.
    `,examples:[{input:"VAE(input_dim=784, latent_dim=20)",output:"Reconstructed images + latent samples",explanation:"Full VAE forward pass"}],starterCode:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        np.random.seed(42)

        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)

        # Latent space (mean and log variance)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Your code here
        pass

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Your code here
        pass

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with mu, log_var, z, x_reconstructed
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        """
        Compute VAE loss (ELBO).

        Returns:
            Dictionary with total_loss, recon_loss, kl_loss
        """
        # Your code here
        pass

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples by decoding random latent vectors.
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Reconstruction shape",input:"recon_shape_check",expected:"True",hidden:!1},{id:"2",description:"KL divergence non-negative",input:"kl_check",expected:"True",hidden:!1}],hints:["Encoder: h = relu(x @ W + b), then mu = h @ W_mu, logvar = h @ W_logvar","Reparameterize: std = exp(0.5 * log_var), z = mu + std * epsilon","KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))","Reconstruction loss: MSE or BCE"],solution:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        np.random.seed(42)

        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)
        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        h = np.maximum(0, x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_recon = 1 / (1 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # Sigmoid
        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {'mu': mu, 'log_var': log_var, 'z': z, 'x_recon': x_recon}

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

        total_loss = recon_loss + kl_loss

        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def sample(self, num_samples: int) -> np.ndarray:
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)
`},{id:"e2e-vqvae",title:"E2E: Vector Quantized VAE",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End VQ-VAE

Implement a complete VQ-VAE (Vector Quantized Variational Autoencoder) with encoder, vector quantization, and decoder.

### Architecture
\`\`\`
Encoder: x → z_e (continuous latents)
           ↓
Vector Quantization: z_e → z_q (discrete latents from codebook)
           ↓
Decoder: z_q → x_reconstructed
\`\`\`

### Key Difference from VAE
- VAE: Continuous latent space with KL regularization
- VQ-VAE: Discrete latent space using learned codebook

### Vector Quantization
1. Encoder outputs continuous vectors z_e
2. Find nearest codebook entry for each spatial position
3. Replace z_e with quantized z_q from codebook
4. Use straight-through estimator for gradients

### Loss Function
\`\`\`
L = L_reconstruction + L_codebook + β * L_commitment

L_reconstruction = ||x - x_hat||²
L_codebook = ||sg[z_e] - e||²  (moves codebook toward encoder output)
L_commitment = ||z_e - sg[e]||²  (commits encoder to codebook)
\`\`\`
Where sg[] is stop-gradient.

### Codebook EMA Update (Alternative)
Instead of gradient updates, codebook can be updated with exponential moving average:
\`\`\`
N_i = γ * N_i + (1 - γ) * n_i     (count of assignments)
m_i = γ * m_i + (1 - γ) * sum(z_e assigned to i)
e_i = m_i / N_i
\`\`\`

### Task
Implement encoder, vector quantization layer, decoder, and all loss components.
    `,examples:[{input:"VQVAE(input_dim=784, num_embeddings=512, embedding_dim=64)",output:"Reconstructed images + discrete codes",explanation:"Full VQ-VAE forward pass"}],starterCode:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Vector Quantization layer.

        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimension of each embedding
        """
        np.random.seed(42)
        # Initialize codebook with small random values
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        """
        Quantize encoder output to nearest codebook entries.

        Args:
            z_e: Encoder output (batch, height, width, embedding_dim)

        Returns:
            z_q: Quantized vectors (same shape as z_e)
            indices: Codebook indices (batch, height, width)
            distances: Distances to nearest embeddings
        """
        # Your code here
        pass

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute VQ losses.

        Args:
            z_e: Encoder output
            z_q: Quantized vectors
            beta: Commitment loss weight

        Returns:
            Dictionary with codebook_loss and commitment_loss
        """
        # Your code here
        pass


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        """
        Initialize VQ-VAE.

        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_embeddings: Codebook size
            embedding_dim: Embedding dimension
        """
        np.random.seed(42)

        # Encoder: conv layers to get spatial feature maps
        self.enc_conv1_w = np.random.randn(hidden_dim, input_channels, 4, 4) * 0.1
        self.enc_conv1_b = np.zeros(hidden_dim)
        self.enc_conv2_w = np.random.randn(embedding_dim, hidden_dim, 4, 4) * 0.1
        self.enc_conv2_b = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: transposed conv to reconstruct
        self.dec_conv1_w = np.random.randn(hidden_dim, embedding_dim, 4, 4) * 0.1
        self.dec_conv1_b = np.zeros(hidden_dim)
        self.dec_conv2_w = np.random.randn(input_channels, hidden_dim, 4, 4) * 0.1
        self.dec_conv2_b = np.zeros(input_channels)

        self.embedding_dim = embedding_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to continuous latent representation.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            z_e: Encoder output (batch, height', width', embedding_dim)
        """
        # Your code here (simplified: use strided conv or just reshape for demo)
        pass

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latents to reconstruction.

        Args:
            z_q: Quantized vectors (batch, height, width, embedding_dim)

        Returns:
            x_recon: Reconstructed input
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with z_e, z_q, indices, x_recon
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute total VQ-VAE loss.

        Returns:
            Dictionary with total_loss, recon_loss, vq_loss, commitment_loss
        """
        # Your code here
        pass

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute codebook usage statistics.

        Returns:
            usage: Count of each codebook entry used
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Quantization shape",input:"vq_shape_check",expected:"True",hidden:!1},{id:"2",description:"Indices valid range",input:"indices_range_check",expected:"True",hidden:!1},{id:"3",description:"Quantized from codebook",input:"quantized_from_codebook_check",expected:"True",hidden:!0},{id:"4",description:"Loss values correct",input:"loss_check",expected:"True",hidden:!0}],hints:["For quantization: flatten spatial dims, compute distances, find argmin","Distance: ||z - e||² = ||z||² + ||e||² - 2*z·e","Straight-through: z_q = z_e + stop_grad(z_q - z_e)","Codebook loss uses z_e detached, commitment uses z_q detached","For simplified encoder/decoder, use reshape operations"],solution:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        np.random.seed(42)
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        # z_e: (batch, H, W, D)
        batch, H, W, D = z_e.shape

        # Flatten spatial dimensions
        z_flat = z_e.reshape(-1, D)  # (N, D) where N = batch * H * W

        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2*z·e
        z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
        e_sq = np.sum(self.embedding ** 2, axis=1)          # (K,)
        cross = z_flat @ self.embedding.T                    # (N, K)
        distances = z_sq + e_sq - 2 * cross                  # (N, K)

        # Find nearest embedding
        indices_flat = np.argmin(distances, axis=1)          # (N,)

        # Get quantized vectors
        z_q_flat = self.embedding[indices_flat]              # (N, D)

        # Reshape back
        z_q = z_q_flat.reshape(batch, H, W, D)
        indices = indices_flat.reshape(batch, H, W)
        min_distances = np.min(distances, axis=1).reshape(batch, H, W)

        return z_q, indices, min_distances

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Codebook loss: ||sg[z_e] - z_q||²
        # In practice, this updates the codebook to move toward encoder output
        codebook_loss = np.mean((z_e - z_q) ** 2)

        # Commitment loss: ||z_e - sg[z_q]||²
        # This commits the encoder output to stay close to codebook
        commitment_loss = beta * np.mean((z_e - z_q) ** 2)

        return {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        np.random.seed(42)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Simplified: use linear projection instead of conv for demo
        # Encoder projects input to embedding space
        self.enc_w1 = np.random.randn(input_channels * 16, hidden_dim) * 0.1
        self.enc_b1 = np.zeros(hidden_dim)
        self.enc_w2 = np.random.randn(hidden_dim, embedding_dim) * 0.1
        self.enc_b2 = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder projects back
        self.dec_w1 = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.dec_b1 = np.zeros(hidden_dim)
        self.dec_w2 = np.random.randn(hidden_dim, input_channels * 16) * 0.1
        self.dec_b2 = np.zeros(input_channels * 16)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, channels, H, W) - assume H=W=4 for simplicity
        batch = x.shape[0]

        # Flatten spatial dimensions
        x_flat = x.reshape(batch, -1)  # (batch, channels * H * W)

        # Encode
        h = np.maximum(0, x_flat @ self.enc_w1 + self.enc_b1)  # ReLU
        z_e = x_flat @ self.enc_w1[:, :self.embedding_dim] + self.enc_b1[:self.embedding_dim]

        # Reshape to spatial format (batch, 2, 2, embedding_dim)
        z_e = z_e.reshape(batch, 1, 1, self.embedding_dim)

        return z_e

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        batch = z_q.shape[0]

        # Flatten
        z_flat = z_q.reshape(batch, -1)  # (batch, embedding_dim)

        # Decode
        h = np.maximum(0, z_flat @ self.dec_w1 + self.dec_b1)  # ReLU
        x_recon = h @ self.dec_w2 + self.dec_b2

        # Reshape to image format
        x_recon = x_recon.reshape(batch, self.input_channels, 4, 4)

        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        # Encode
        z_e = self.encode(x)

        # Quantize
        z_q, indices, distances = self.vq.quantize(z_e)

        # Straight-through estimator: gradient flows through z_q to z_e
        # z_q_st = z_e + (z_q - z_e).detach()
        # For forward pass, we just use z_q

        # Decode
        x_recon = self.decode(z_q)

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon
        }

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Reconstruction loss
        recon_loss = np.mean((x - x_recon) ** 2)

        # VQ losses
        vq_losses = self.vq.compute_loss(z_e, z_q, beta)

        # Total loss
        total_loss = recon_loss + vq_losses['codebook_loss'] + vq_losses['commitment_loss']

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss']
        }

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        usage = np.bincount(indices.flatten(), minlength=self.vq.num_embeddings)
        return usage
`},{id:"e2e-diffusion",title:"E2E: Diffusion Model",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Diffusion Model

Implement core components of a diffusion model (DDPM-style).

### Forward Process (Adding Noise)
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### Noise Schedule
\`\`\`
β_t: Linear from β_start to β_end
α_t = 1 - β_t
α_bar_t = cumulative product of α_t
\`\`\`

### Training Objective
\`\`\`
L = ||ε - ε_θ(x_t, t)||²
\`\`\`

Model learns to predict the noise added at timestep t.

### Reverse Process (Sampling)
\`\`\`
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z
\`\`\`

### Task
Implement noise schedule, forward process, and reverse sampling.
    `,examples:[{input:"DiffusionModel(T=1000)",output:"Denoised samples",explanation:"Forward and reverse diffusion"}],starterCode:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize diffusion model.

        Args:
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.T = T

        # Compute noise schedule
        # Your code here
        pass

    def get_noise_schedule(self) -> dict:
        """Return computed noise schedule parameters."""
        # Your code here
        pass

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        """
        Add noise to data at timestep t.

        Args:
            x_0: Original data (batch, ...)
            t: Timestep

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        np.random.seed(42)
        # Your code here
        pass

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by model

        Returns:
            x_{t-1}: Less noisy data
        """
        # Your code here
        pass

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        """Compute MSE loss between true and predicted noise."""
        # Your code here
        pass

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        """
        Generate samples using reverse diffusion.

        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)

        Returns:
            Generated samples
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Alpha bar decreases",input:"alpha_bar_check",expected:"True",hidden:!1},{id:"2",description:"Forward noise shape",input:"forward_check",expected:"True",hidden:!1}],hints:["betas = np.linspace(beta_start, beta_end, T)","alpha_bars = np.cumprod(1 - betas)","x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise","Reverse: use mean and add small noise (except at t=0)"],solution:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = T

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # For reverse process
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1 - self.alpha_bars)
        self.sqrt_alphas = np.sqrt(self.alphas)

    def get_noise_schedule(self) -> dict:
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars
        }

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        np.random.seed(42)
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Mean of reverse distribution
        coef1 = 1 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            # Add noise (except at final step)
            sigma = np.sqrt(beta)
            noise = np.random.randn(*x_t.shape)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        return np.mean((true_noise - predicted_noise) ** 2)

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        # Start from pure noise
        x = np.random.randn(*shape)

        # Reverse diffusion
        for t in reversed(range(self.T)):
            predicted_noise = noise_predictor(x, t)
            x = self.reverse_step(x, t, predicted_noise)

        return x
`},{id:"e2e-cnn",title:"E2E: Convolutional Neural Network",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End CNN

Implement a complete CNN for image classification.

### Architecture
\`\`\`
Input (batch, channels, height, width)
  ↓
Conv2D(in_channels, 32, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Conv2D(32, 64, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Flatten
  ↓
Linear(flattened_size, 128) + ReLU
  ↓
Linear(128, num_classes)
  ↓
Softmax
\`\`\`

### Components
- 2D Convolution (no padding for simplicity)
- Max Pooling
- Flatten
- Fully Connected layers

### Task
Implement conv2d, maxpool, and combine into CNN.
    `,examples:[{input:"CNN on 28x28 grayscale images",output:"Class probabilities",explanation:"Full CNN forward pass"}],starterCode:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    2D convolution (no padding, stride=1).

    Args:
        x: Input (batch, in_channels, height, width)
        kernel: Weights (out_channels, in_channels, kH, kW)
        bias: Bias (out_channels,)

    Returns:
        output: (batch, out_channels, out_height, out_width)
    """
    # Your code here
    pass


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2D max pooling.

    Args:
        x: Input (batch, channels, height, width)
        pool_size: Size of pooling window

    Returns:
        output: Pooled tensor
    """
    # Your code here
    pass


def flatten(x: np.ndarray) -> np.ndarray:
    """Flatten all dimensions except batch."""
    # Your code here
    pass


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layer 1: in_channels -> 32
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)

        # Conv layer 2: 32 -> 64
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate flattened size after convolutions and pooling
        # Your code to compute this
        pass

        # Fully connected layers
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        # Your code here
        pass


def test_cnn():
    """Test CNN with sample data."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Conv output shape",input:"conv_shape_check",expected:"True",hidden:!1},{id:"2",description:"Pool output shape",input:"pool_shape_check",expected:"True",hidden:!1}],hints:["Conv output size: (input - kernel + 1) for stride=1, no padding","Pool output size: input // pool_size","Use nested loops for convolution (or np.lib.stride_tricks)","Flatten: x.reshape(batch_size, -1)"],solution:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, in_ch, H, W = x.shape
    out_ch, _, kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((batch, out_ch, out_H, out_W))

    for b in range(batch):
        for oc in range(out_ch):
            for i in range(out_H):
                for j in range(out_W):
                    receptive_field = x[b, :, i:i+kH, j:j+kW]
                    output[b, oc, i, j] = np.sum(receptive_field * kernel[oc]) + bias[oc]

    return output


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    batch, channels, H, W = x.shape
    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((batch, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * pool_size
            w_start = j * pool_size
            output[:, :, i, j] = np.max(
                x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size],
                axis=(2, 3)
            )

    return output


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layers
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate size after convolutions and pooling
        # After conv1 (3x3): H-2, W-2
        # After pool1 (2x2): (H-2)//2, (W-2)//2
        # After conv2 (3x3): (H-2)//2 - 2, (W-2)//2 - 2
        # After pool2 (2x2): ((H-2)//2 - 2)//2, ((W-2)//2 - 2)//2
        h1 = (H - 2) // 2
        w1 = (W - 2) // 2
        h2 = (h1 - 2) // 2
        w2 = (w1 - 2) // 2
        flat_size = 64 * h2 * w2

        # FC layers
        self.fc1_w = np.random.randn(flat_size, 128) * 0.1
        self.fc1_b = np.zeros(128)
        self.fc2_w = np.random.randn(128, num_classes) * 0.1
        self.fc2_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Conv block 1
        x = conv2d(x, self.conv1_w, self.conv1_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Conv block 2
        x = conv2d(x, self.conv2_w, self.conv2_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Flatten
        x = flatten(x)

        # FC layers
        x = np.maximum(0, x @ self.fc1_w + self.fc1_b)  # ReLU
        x = x @ self.fc2_w + self.fc2_b  # Logits

        return x


def test_cnn():
    # Test with MNIST-like data
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    x = np.random.randn(4, 1, 28, 28)
    logits = cnn.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    return logits.shape == (4, 10)
`}],dh=[...ek,...tk,...nk,...rk,...ik,...ak,...ok,...sk,...lk,...uk,...ck,...dk,...pk,...fk];function mk(e){return dh.find(t=>t.id===e)}function hk(e){return dh.filter(t=>t.section===e)}function gk(){const{sectionId:e}=km(),{getProblemProgress:t,getSectionProgress:n}=Mi(),r=Sr.find(o=>o.id===e),i=e?hk(e):[];if(!r)return x.jsxs("div",{className:"text-center py-12",children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Section Not Found"}),x.jsx(yn,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const a=n(r.id,r.problems.length);return x.jsxs("div",{className:"max-w-4xl mx-auto",children:[x.jsxs("div",{className:"mb-8",children:[x.jsxs("div",{className:"flex items-center gap-4 mb-4",children:[x.jsx("div",{className:"w-14 h-14 bg-white border border-gray-200 rounded-xl flex items-center justify-center text-3xl shadow-sm",children:r.icon}),x.jsxs("div",{children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900",children:r.title}),x.jsx("p",{className:"text-gray-500",children:r.description})]})]}),x.jsxs("div",{className:"flex items-center gap-4",children:[x.jsx("div",{className:"flex-1 h-2 bg-gray-200 rounded-full overflow-hidden",children:x.jsx("div",{className:"h-full bg-primary-500 transition-all duration-300",style:{width:`${a}%`}})}),x.jsxs("span",{className:"text-primary-600 font-medium",children:[a,"% complete"]})]})]}),x.jsxs("div",{className:"bg-white rounded-xl p-6 mb-8 border border-gray-200 shadow-sm",children:[x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Introduction"}),x.jsx("div",{className:"prose prose-sm max-w-none",children:x.jsx(ch,{components:{h1:({children:o})=>x.jsx("h1",{className:"text-xl font-bold text-gray-900 mt-0 mb-4",children:o}),h2:({children:o})=>x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:o}),h3:({children:o})=>x.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:o}),p:({children:o})=>x.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:o}),code:({className:o,children:s})=>(o==null?void 0:o.includes("language-"))?x.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:x.jsx("code",{className:"text-sm text-gray-800",children:s})}):x.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm",children:s}),ul:({children:o})=>x.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:o}),li:({children:o})=>x.jsx("li",{className:"text-gray-600",children:o}),strong:({children:o})=>x.jsx("strong",{className:"text-gray-900 font-semibold",children:o})},children:r.introduction})})]}),x.jsxs("div",{children:[x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Problems"}),x.jsx("div",{className:"space-y-3",children:i.map((o,s)=>{const l=t(r.id,o.id);return x.jsxs(yn,{to:`/problem/${r.id}/${o.id}`,className:"flex items-center gap-4 bg-white rounded-lg p-4 border border-gray-200 hover:border-primary-400 hover:shadow-sm transition-all group",children:[x.jsx("div",{className:`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${l.status==="completed"?"bg-green-100 text-green-600":l.status==="in_progress"?"bg-yellow-100 text-yellow-600":"bg-gray-100 text-gray-500"}`,children:l.status==="completed"?"✓":s+1}),x.jsxs("div",{className:"flex-1",children:[x.jsx("h3",{className:"text-gray-900 font-medium group-hover:text-primary-600 transition-colors",children:o.title}),x.jsxs("div",{className:"flex items-center gap-3 mt-1",children:[x.jsx("span",{className:`text-xs px-2 py-0.5 rounded ${o.difficulty==="easy"?"bg-green-100 text-green-700":o.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:o.difficulty}),x.jsxs("span",{className:"text-gray-400 text-xs",children:[o.testCases.length," test cases"]})]})]}),x.jsx("div",{className:"text-gray-400 group-hover:text-primary-600 transition-colors",children:"→"})]},o.id)})})]})]})}var ph={exports:{}},yk="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED",xk=yk,_k=xk;function fh(){}function mh(){}mh.resetWarningCache=fh;var vk=function(){function e(r,i,a,o,s,l){if(l!==_k){var u=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw u.name="Invariant Violation",u}}e.isRequired=e;function t(){return e}var n={array:e,bigint:e,bool:e,func:e,number:e,object:e,string:e,symbol:e,any:e,arrayOf:t,element:e,elementType:e,instanceOf:t,node:e,objectOf:t,oneOf:t,oneOfType:t,shape:t,exact:t,checkPropTypes:mh,resetWarningCache:fh};return n.PropTypes=n,n};ph.exports=vk();var wk=ph.exports;const ae=ro(wk);var ut=typeof window<"u"?window:null,Mu=ut===null,Pi=Mu?void 0:ut.document,_t="addEventListener",vt="removeEventListener",ms="getBoundingClientRect",Ur="_a",wt="_b",Bt="_c",sa="horizontal",kt=function(){return!1},kk=Mu?"calc":["","-webkit-","-moz-","-o-"].filter(function(e){var t=Pi.createElement("div");return t.style.cssText="width:"+e+"calc(9px)",!!t.style.length}).shift()+"calc",hh=function(e){return typeof e=="string"||e instanceof String},Qd=function(e){if(hh(e)){var t=Pi.querySelector(e);if(!t)throw new Error("Selector "+e+" did not match a DOM element");return t}return e},je=function(e,t,n){var r=e[t];return r!==void 0?r:n},la=function(e,t,n,r){if(t){if(r==="end")return 0;if(r==="center")return e/2}else if(n){if(r==="start")return 0;if(r==="center")return e/2}return e},bk=function(e,t){var n=Pi.createElement("div");return n.className="gutter gutter-"+t,n},Sk=function(e,t,n){var r={};return hh(t)?r[e]=t:r[e]=kk+"("+t+"% - "+n+"px)",r},Ck=function(e,t){var n;return n={},n[e]=t+"px",n},Xd=function(e,t){if(t===void 0&&(t={}),Mu)return{};var n=e,r,i,a,o,s,l;Array.from&&(n=Array.from(n));var u=Qd(n[0]),c=u.parentNode,d=getComputedStyle?getComputedStyle(c):null,p=d?d.flexDirection:null,f=je(t,"sizes")||n.map(function(){return 100/n.length}),w=je(t,"minSize",100),_=Array.isArray(w)?w:n.map(function(){return w}),S=je(t,"maxSize",1/0),h=Array.isArray(S)?S:n.map(function(){return S}),m=je(t,"expandToMin",!1),g=je(t,"gutterSize",10),b=je(t,"gutterAlign","center"),z=je(t,"snapOffset",30),k=Array.isArray(z)?z:n.map(function(){return z}),E=je(t,"dragInterval",1),N=je(t,"direction",sa),O=je(t,"cursor",N===sa?"col-resize":"row-resize"),L=je(t,"gutter",bk),F=je(t,"elementStyle",Sk),U=je(t,"gutterStyle",Ck);N===sa?(r="width",i="clientX",a="left",o="right",s="clientWidth"):N==="vertical"&&(r="height",i="clientY",a="top",o="bottom",s="clientHeight");function Y(W,T,M,H){var Ce=F(r,T,M,H);Object.keys(Ce).forEach(function(xe){W.style[xe]=Ce[xe]})}function re(W,T,M){var H=U(r,T,M);Object.keys(H).forEach(function(Ce){W.style[Ce]=H[Ce]})}function j(){return l.map(function(W){return W.size})}function Q(W){return"touches"in W?W.touches[0][i]:W[i]}function ee(W){var T=l[this.a],M=l[this.b],H=T.size+M.size;T.size=W/this.size*H,M.size=H-W/this.size*H,Y(T.element,T.size,this[wt],T.i),Y(M.element,M.size,this[Bt],M.i)}function I(W){var T,M=l[this.a],H=l[this.b];this.dragging&&(T=Q(W)-this.start+(this[wt]-this.dragOffset),E>1&&(T=Math.round(T/E)*E),T<=M.minSize+M.snapOffset+this[wt]?T=M.minSize+this[wt]:T>=this.size-(H.minSize+H.snapOffset+this[Bt])&&(T=this.size-(H.minSize+this[Bt])),T>=M.maxSize-M.snapOffset+this[wt]?T=M.maxSize+this[wt]:T<=this.size-(H.maxSize-H.snapOffset+this[Bt])&&(T=this.size-(H.maxSize+this[Bt])),ee.call(this,T),je(t,"onDrag",kt)(j()))}function V(){var W=l[this.a].element,T=l[this.b].element,M=W[ms](),H=T[ms]();this.size=M[r]+H[r]+this[wt]+this[Bt],this.start=M[a],this.end=M[o]}function y(W){if(!getComputedStyle)return null;var T=getComputedStyle(W);if(!T)return null;var M=W[s];return M===0?null:(N===sa?M-=parseFloat(T.paddingLeft)+parseFloat(T.paddingRight):M-=parseFloat(T.paddingTop)+parseFloat(T.paddingBottom),M)}function B(W){var T=y(c);if(T===null||_.reduce(function(xe,Fe){return xe+Fe},0)>T)return W;var M=0,H=[],Ce=W.map(function(xe,Fe){var Nt=T*xe/100,En=la(g,Fe===0,Fe===W.length-1,b),Pn=_[Fe]+En;return Nt<Pn?(M+=Pn-Nt,H.push(0),Pn):(H.push(Nt-Pn),Nt)});return M===0?W:Ce.map(function(xe,Fe){var Nt=xe;if(M>0&&H[Fe]-M>0){var En=Math.min(M,H[Fe]-M);M-=En,Nt=xe-En}return Nt/T*100})}function $(){var W=this,T=l[W.a].element,M=l[W.b].element;W.dragging&&je(t,"onDragEnd",kt)(j()),W.dragging=!1,ut[vt]("mouseup",W.stop),ut[vt]("touchend",W.stop),ut[vt]("touchcancel",W.stop),ut[vt]("mousemove",W.move),ut[vt]("touchmove",W.move),W.stop=null,W.move=null,T[vt]("selectstart",kt),T[vt]("dragstart",kt),M[vt]("selectstart",kt),M[vt]("dragstart",kt),T.style.userSelect="",T.style.webkitUserSelect="",T.style.MozUserSelect="",T.style.pointerEvents="",M.style.userSelect="",M.style.webkitUserSelect="",M.style.MozUserSelect="",M.style.pointerEvents="",W.gutter.style.cursor="",W.parent.style.cursor="",Pi.body.style.cursor=""}function v(W){if(!("button"in W&&W.button!==0)){var T=this,M=l[T.a].element,H=l[T.b].element;T.dragging||je(t,"onDragStart",kt)(j()),W.preventDefault(),T.dragging=!0,T.move=I.bind(T),T.stop=$.bind(T),ut[_t]("mouseup",T.stop),ut[_t]("touchend",T.stop),ut[_t]("touchcancel",T.stop),ut[_t]("mousemove",T.move),ut[_t]("touchmove",T.move),M[_t]("selectstart",kt),M[_t]("dragstart",kt),H[_t]("selectstart",kt),H[_t]("dragstart",kt),M.style.userSelect="none",M.style.webkitUserSelect="none",M.style.MozUserSelect="none",M.style.pointerEvents="none",H.style.userSelect="none",H.style.webkitUserSelect="none",H.style.MozUserSelect="none",H.style.pointerEvents="none",T.gutter.style.cursor=O,T.parent.style.cursor=O,Pi.body.style.cursor=O,V.call(T),T.dragOffset=Q(W)-T.end}}f=B(f);var te=[];l=n.map(function(W,T){var M={element:Qd(W),size:f[T],minSize:_[T],maxSize:h[T],snapOffset:k[T],i:T},H;if(T>0&&(H={a:T-1,b:T,dragging:!1,direction:N,parent:c},H[wt]=la(g,T-1===0,!1,b),H[Bt]=la(g,!1,T===n.length-1,b),p==="row-reverse"||p==="column-reverse")){var Ce=H.a;H.a=H.b,H.b=Ce}if(T>0){var xe=L(T,N,M.element);re(xe,g,T),H[Ur]=v.bind(H),xe[_t]("mousedown",H[Ur]),xe[_t]("touchstart",H[Ur]),c.insertBefore(xe,M.element),H.gutter=xe}return Y(M.element,M.size,la(g,T===0,T===n.length-1,b),T),T>0&&te.push(H),M});function we(W){var T=W.i===te.length,M=T?te[W.i-1]:te[W.i];V.call(M);var H=T?M.size-W.minSize-M[Bt]:W.minSize+M[wt];ee.call(M,H)}l.forEach(function(W){var T=W.element[ms]()[r];T<W.minSize&&(m?we(W):W.minSize=T)});function se(W){var T=B(W);T.forEach(function(M,H){if(H>0){var Ce=te[H-1],xe=l[Ce.a],Fe=l[Ce.b];xe.size=T[H-1],Fe.size=M,Y(xe.element,xe.size,Ce[wt],xe.i),Y(Fe.element,Fe.size,Ce[Bt],Fe.i)}})}function lt(W,T){te.forEach(function(M){if(T!==!0?M.parent.removeChild(M.gutter):(M.gutter[vt]("mousedown",M[Ur]),M.gutter[vt]("touchstart",M[Ur])),W!==!0){var H=F(r,M.a.size,M[wt]);Object.keys(H).forEach(function(Ce){l[M.a].element.style[Ce]="",l[M.b].element.style[Ce]=""})}})}return{setSizes:se,getSizes:j,collapse:function(T){we(l[T])},destroy:lt,parent:c,pairs:te}};function hs(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)===-1&&(n[r]=e[r]);return n}var to=function(e){function t(){e.apply(this,arguments)}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.componentDidMount=function(){var r=this.props;r.children;var i=r.gutter,a=hs(r,["children","gutter"]),o=a;o.gutter=function(s,l){var u;return i?u=i(s,l):(u=document.createElement("div"),u.className="gutter gutter-"+l),u.__isSplitGutter=!0,u},this.split=Xd(this.parent.children,o)},t.prototype.componentDidUpdate=function(r){var i=this,a=this.props;a.children;var o=a.minSize,s=a.sizes,l=a.collapsed,u=hs(a,["children","minSize","sizes","collapsed"]),c=u,d=r.minSize,p=r.sizes,f=r.collapsed,w=["maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor"],_=w.map(function(m){return i.props[m]!==r[m]}).reduce(function(m,g){return m||g},!1);if(Array.isArray(o)&&Array.isArray(d)){var S=!1;o.forEach(function(m,g){S=S||m!==d[g]}),_=_||S}else Array.isArray(o)||Array.isArray(d)?_=!0:_=_||o!==d;if(_)c.minSize=o,c.sizes=s||this.split.getSizes(),this.split.destroy(!0,!0),c.gutter=function(m,g,b){return b.previousSibling},this.split=Xd(Array.from(this.parent.children).filter(function(m){return!m.__isSplitGutter}),c);else if(s){var h=!1;s.forEach(function(m,g){h=h||m!==p[g]}),h&&this.split.setSizes(this.props.sizes)}Number.isInteger(l)&&(l!==f||_)&&this.split.collapse(l)},t.prototype.componentWillUnmount=function(){this.split.destroy(),delete this.split},t.prototype.render=function(){var r=this,i=this.props;i.sizes,i.minSize,i.maxSize,i.expandToMin,i.gutterSize,i.gutterAlign,i.snapOffset,i.dragInterval,i.direction,i.cursor,i.gutter,i.elementStyle,i.gutterStyle,i.onDrag,i.onDragStart,i.onDragEnd,i.collapsed;var a=i.children,o=hs(i,["sizes","minSize","maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor","gutter","elementStyle","gutterStyle","onDrag","onDragStart","onDragEnd","collapsed","children"]),s=o;return jt.createElement("div",Object.assign({},{ref:function(l){r.parent=l}},s),a)},t}(jt.Component);to.propTypes={sizes:ae.arrayOf(ae.number),minSize:ae.oneOfType([ae.number,ae.arrayOf(ae.number)]),maxSize:ae.oneOfType([ae.number,ae.arrayOf(ae.number)]),expandToMin:ae.bool,gutterSize:ae.number,gutterAlign:ae.string,snapOffset:ae.oneOfType([ae.number,ae.arrayOf(ae.number)]),dragInterval:ae.number,direction:ae.string,cursor:ae.string,gutter:ae.func,elementStyle:ae.func,gutterStyle:ae.func,onDrag:ae.func,onDragStart:ae.func,onDragEnd:ae.func,collapsed:ae.number,children:ae.arrayOf(ae.element)};to.defaultProps={sizes:void 0,minSize:void 0,maxSize:void 0,expandToMin:void 0,gutterSize:void 0,gutterAlign:void 0,snapOffset:void 0,dragInterval:void 0,direction:void 0,cursor:void 0,gutter:void 0,elementStyle:void 0,gutterStyle:void 0,onDrag:void 0,onDragStart:void 0,onDragEnd:void 0,collapsed:void 0,children:void 0};function zk(){const[e,t]=C.useState(!1),[n,r]=C.useState(!1),[i,a]=C.useState(null),[o,s]=C.useState([]),l=C.useRef(null),u=C.useRef(!1),c=C.useCallback(_=>{s(S=>[...S,_])},[]),d=C.useCallback(()=>{s([])},[]),p=C.useCallback(async()=>{if(!(l.current||u.current)){u.current=!0,t(!0),a(null);try{window.loadPyodide||await new Promise((S,h)=>{const m=document.createElement("script");m.src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js",m.async=!0,m.onload=()=>S(),m.onerror=()=>h(new Error("Failed to load Pyodide")),document.head.appendChild(m)});const _=await window.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"});_.setStdout({batched:S=>{S.trim()&&c(S)}}),_.setStderr({batched:S=>{S.trim()&&c(`[Error] ${S}`)}}),await _.loadPackage(["numpy"]),l.current=_,r(!0),c("Python environment ready with NumPy loaded.")}catch(_){const S=_ instanceof Error?_.message:"Failed to initialize Python";a(S),c(`[Error] ${S}`)}finally{t(!1),u.current=!1}}},[c]);C.useEffect(()=>{p()},[p]);const f=C.useCallback(async _=>{if(!l.current)return{success:!1,error:"Python environment not ready"};try{return d(),await l.current.runPythonAsync(_),{success:!0}}catch(S){const h=S instanceof Error?S.message:"Execution error";return c(`[Error] ${h}`),{success:!1,error:h}}},[c,d]),w=C.useCallback(async(_,S,h)=>{if(!l.current)return S.map(g=>({id:g.id,passed:!1,description:g.description,expected:g.expected,actual:"Python environment not ready",hidden:g.hidden}));const m=[];d();try{await l.current.runPythonAsync(_);for(const g of S)try{const b=`
import numpy as np
import json

# Parse input
test_input = ${g.input}

# Call the function
if isinstance(test_input, list):
    result = ${h}(np.array(test_input))
elif isinstance(test_input, tuple):
    result = ${h}(*[np.array(x) if isinstance(x, list) else x for x in test_input])
else:
    result = ${h}(test_input)

# Convert result to string for comparison
if isinstance(result, np.ndarray):
    result_str = str(result.tolist())
elif isinstance(result, (list, dict)):
    result_str = json.dumps(result)
elif isinstance(result, float):
    result_str = str(round(result, 6))
else:
    result_str = str(result)
result_str
`,z=await l.current.runPythonAsync(b),k=String(z),E=g.expected.replace(/\s/g,""),N=k.replace(/\s/g,""),O=E===N;m.push({id:g.id,passed:O,description:g.description,expected:g.expected,actual:k,hidden:g.hidden}),c(O?`Test ${g.id}: PASSED`:`Test ${g.id}: FAILED - Expected ${g.expected}, got ${k}`)}catch(b){const z=b instanceof Error?b.message:"Test execution error";m.push({id:g.id,passed:!1,description:g.description,expected:g.expected,actual:z,hidden:g.hidden}),c(`Test ${g.id}: ERROR - ${z}`)}}catch(g){const b=g instanceof Error?g.message:"Code execution error";return c(`[Error] ${b}`),S.map(z=>({id:z.id,passed:!1,description:z.description,expected:z.expected,actual:b,hidden:z.hidden}))}return m},[c,d]);return{isLoading:e,isReady:n,error:i,output:o,runCode:f,runTests:w,clearOutput:d}}function Kd(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function Ek(e){if(Array.isArray(e))return e}function Pk(e,t,n){return(t=jk(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Nk(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,i,a,o,s=[],l=!0,u=!1;try{if(a=(n=n.call(e)).next,t!==0)for(;!(l=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);l=!0);}catch(c){u=!0,i=c}finally{try{if(!l&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(u)throw i}}return s}}function Tk(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function Yd(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function Gd(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?Yd(Object(n),!0).forEach(function(r){Pk(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Yd(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function Lk(e,t){if(e==null)return{};var n,r,i=Ak(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}function Ak(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function Rk(e,t){return Ek(e)||Nk(e,t)||Ok(e,t)||Tk()}function Ik(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function jk(e){var t=Ik(e,"string");return typeof t=="symbol"?t:t+""}function Ok(e,t){if(e){if(typeof e=="string")return Kd(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?Kd(e,t):void 0}}function Mk(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Zd(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function Jd(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?Zd(Object(n),!0).forEach(function(r){Mk(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Zd(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function Dk(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(i,a){return a(i)},r)}}function Xr(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function no(e){return{}.toString.call(e).includes("Object")}function Fk(e){return!Object.keys(e).length}function Ni(e){return typeof e=="function"}function Wk(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function Bk(e,t){return no(t)||xn("changeType"),Object.keys(t).some(function(n){return!Wk(e,n)})&&xn("changeField"),t}function qk(e){Ni(e)||xn("selectorType")}function Uk(e){Ni(e)||no(e)||xn("handlerType"),no(e)&&Object.values(e).some(function(t){return!Ni(t)})&&xn("handlersType")}function Vk(e){e||xn("initialIsRequired"),no(e)||xn("initialType"),Fk(e)&&xn("initialContent")}function Hk(e,t){throw new Error(e[t]||e.default)}var $k={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},xn=Xr(Hk)($k),ua={changes:Bk,selector:qk,handler:Uk,initial:Vk};function Qk(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};ua.initial(e),ua.handler(t);var n={current:e},r=Xr(Yk)(n,t),i=Xr(Kk)(n),a=Xr(ua.changes)(e),o=Xr(Xk)(n);function s(){var u=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(c){return c};return ua.selector(u),u(n.current)}function l(u){Dk(r,i,a,o)(u)}return[s,l]}function Xk(e,t){return Ni(t)?t(e.current):t}function Kk(e,t){return e.current=Jd(Jd({},e.current),t),t}function Yk(e,t,n){return Ni(t)?t(e.current):Object.keys(n).forEach(function(r){var i;return(i=t[r])===null||i===void 0?void 0:i.call(t,e.current[r])}),n}var Gk={create:Qk},Zk={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function Jk(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),a=0;a<r;a++)i[a]=arguments[a];return i.length>=e.length?e.apply(this,i):function(){for(var o=arguments.length,s=new Array(o),l=0;l<o;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function eb(e){return{}.toString.call(e).includes("Object")}function tb(e){return e||ep("configIsRequired"),eb(e)||ep("configType"),e.urls?(nb(),{paths:{vs:e.urls.monacoBase}}):e}function nb(){console.warn(gh.deprecation)}function rb(e,t){throw new Error(e[t]||e.default)}var gh={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},ep=Jk(rb)(gh),ib={config:tb},ab=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(i){return n.reduceRight(function(a,o){return o(a)},i)}};function yh(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],yh(e[n],t[n]))}),Gd(Gd({},e),t)}var ob={type:"cancelation",msg:"operation is manually canceled"};function gs(e){var t=!1,n=new Promise(function(r,i){e.then(function(a){return t?i(ob):r(a)}),e.catch(i)});return n.cancel=function(){return t=!0},n}var sb=["monaco"],lb=Gk.create({config:Zk,isInitialized:!1,resolve:null,reject:null,monaco:null}),xh=Rk(lb,2),Wi=xh[0],Po=xh[1];function ub(e){var t=ib.config(e),n=t.monaco,r=Lk(t,sb);Po(function(i){return{config:yh(i.config,r),monaco:n}})}function cb(){var e=Wi(function(t){var n=t.monaco,r=t.isInitialized,i=t.resolve;return{monaco:n,isInitialized:r,resolve:i}});if(!e.isInitialized){if(Po({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),gs(ys);if(window.monaco&&window.monaco.editor)return _h(window.monaco),e.resolve(window.monaco),gs(ys);ab(db,fb)(mb)}return gs(ys)}function db(e){return document.body.appendChild(e)}function pb(e){var t=document.createElement("script");return e&&(t.src=e),t}function fb(e){var t=Wi(function(r){var i=r.config,a=r.reject;return{config:i,reject:a}}),n=pb("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function mb(){var e=Wi(function(n){var r=n.config,i=n.resolve,a=n.reject;return{config:r,resolve:i,reject:a}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;_h(r),e.resolve(r)},function(n){e.reject(n)})}function _h(e){Wi().monaco||Po({monaco:e})}function hb(){return Wi(function(e){var t=e.monaco;return t})}var ys=new Promise(function(e,t){return Po({resolve:e,reject:t})}),vh={config:ub,init:cb,__getMonacoInstance:hb},gb={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},xs=gb,yb={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},xb=yb;function _b({children:e}){return jt.createElement("div",{style:xb.container},e)}var vb=_b,wb=vb;function kb({width:e,height:t,isEditorReady:n,loading:r,_ref:i,className:a,wrapperProps:o}){return jt.createElement("section",{style:{...xs.wrapper,width:e,height:t},...o},!n&&jt.createElement(wb,null,r),jt.createElement("div",{ref:i,style:{...xs.fullWidth,...!n&&xs.hide},className:a}))}var bb=kb,wh=C.memo(bb);function Sb(e){C.useEffect(e,[])}var kh=Sb;function Cb(e,t,n=!0){let r=C.useRef(!0);C.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var tt=Cb;function si(){}function sr(e,t,n,r){return zb(e,r)||Eb(e,t,n,r)}function zb(e,t){return e.editor.getModel(bh(e,t))}function Eb(e,t,n,r){return e.editor.createModel(t,n,r?bh(e,r):void 0)}function bh(e,t){return e.Uri.parse(t)}function Pb({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:i,originalModelPath:a,modifiedModelPath:o,keepCurrentOriginalModel:s=!1,keepCurrentModifiedModel:l=!1,theme:u="light",loading:c="Loading...",options:d={},height:p="100%",width:f="100%",className:w,wrapperProps:_={},beforeMount:S=si,onMount:h=si}){let[m,g]=C.useState(!1),[b,z]=C.useState(!0),k=C.useRef(null),E=C.useRef(null),N=C.useRef(null),O=C.useRef(h),L=C.useRef(S),F=C.useRef(!1);kh(()=>{let j=vh.init();return j.then(Q=>(E.current=Q)&&z(!1)).catch(Q=>(Q==null?void 0:Q.type)!=="cancelation"&&console.error("Monaco initialization: error:",Q)),()=>k.current?re():j.cancel()}),tt(()=>{if(k.current&&E.current){let j=k.current.getOriginalEditor(),Q=sr(E.current,e||"",r||n||"text",a||"");Q!==j.getModel()&&j.setModel(Q)}},[a],m),tt(()=>{if(k.current&&E.current){let j=k.current.getModifiedEditor(),Q=sr(E.current,t||"",i||n||"text",o||"");Q!==j.getModel()&&j.setModel(Q)}},[o],m),tt(()=>{let j=k.current.getModifiedEditor();j.getOption(E.current.editor.EditorOption.readOnly)?j.setValue(t||""):t!==j.getValue()&&(j.executeEdits("",[{range:j.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),j.pushUndoStop())},[t],m),tt(()=>{var j,Q;(Q=(j=k.current)==null?void 0:j.getModel())==null||Q.original.setValue(e||"")},[e],m),tt(()=>{let{original:j,modified:Q}=k.current.getModel();E.current.editor.setModelLanguage(j,r||n||"text"),E.current.editor.setModelLanguage(Q,i||n||"text")},[n,r,i],m),tt(()=>{var j;(j=E.current)==null||j.editor.setTheme(u)},[u],m),tt(()=>{var j;(j=k.current)==null||j.updateOptions(d)},[d],m);let U=C.useCallback(()=>{var ee;if(!E.current)return;L.current(E.current);let j=sr(E.current,e||"",r||n||"text",a||""),Q=sr(E.current,t||"",i||n||"text",o||"");(ee=k.current)==null||ee.setModel({original:j,modified:Q})},[n,t,i,e,r,a,o]),Y=C.useCallback(()=>{var j;!F.current&&N.current&&(k.current=E.current.editor.createDiffEditor(N.current,{automaticLayout:!0,...d}),U(),(j=E.current)==null||j.editor.setTheme(u),g(!0),F.current=!0)},[d,u,U]);C.useEffect(()=>{m&&O.current(k.current,E.current)},[m]),C.useEffect(()=>{!b&&!m&&Y()},[b,m,Y]);function re(){var Q,ee,I,V;let j=(Q=k.current)==null?void 0:Q.getModel();s||((ee=j==null?void 0:j.original)==null||ee.dispose()),l||((I=j==null?void 0:j.modified)==null||I.dispose()),(V=k.current)==null||V.dispose()}return jt.createElement(wh,{width:f,height:p,isEditorReady:m,loading:c,_ref:N,className:w,wrapperProps:_})}var Nb=Pb;C.memo(Nb);function Tb(e){let t=C.useRef();return C.useEffect(()=>{t.current=e},[e]),t.current}var Lb=Tb,ca=new Map;function Ab({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:i,path:a,theme:o="light",line:s,loading:l="Loading...",options:u={},overrideServices:c={},saveViewState:d=!0,keepCurrentModel:p=!1,width:f="100%",height:w="100%",className:_,wrapperProps:S={},beforeMount:h=si,onMount:m=si,onChange:g,onValidate:b=si}){let[z,k]=C.useState(!1),[E,N]=C.useState(!0),O=C.useRef(null),L=C.useRef(null),F=C.useRef(null),U=C.useRef(m),Y=C.useRef(h),re=C.useRef(),j=C.useRef(r),Q=Lb(a),ee=C.useRef(!1),I=C.useRef(!1);kh(()=>{let B=vh.init();return B.then($=>(O.current=$)&&N(!1)).catch($=>($==null?void 0:$.type)!=="cancelation"&&console.error("Monaco initialization: error:",$)),()=>L.current?y():B.cancel()}),tt(()=>{var $,v,te,we;let B=sr(O.current,e||r||"",t||i||"",a||n||"");B!==(($=L.current)==null?void 0:$.getModel())&&(d&&ca.set(Q,(v=L.current)==null?void 0:v.saveViewState()),(te=L.current)==null||te.setModel(B),d&&((we=L.current)==null||we.restoreViewState(ca.get(a))))},[a],z),tt(()=>{var B;(B=L.current)==null||B.updateOptions(u)},[u],z),tt(()=>{!L.current||r===void 0||(L.current.getOption(O.current.editor.EditorOption.readOnly)?L.current.setValue(r):r!==L.current.getValue()&&(I.current=!0,L.current.executeEdits("",[{range:L.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),L.current.pushUndoStop(),I.current=!1))},[r],z),tt(()=>{var $,v;let B=($=L.current)==null?void 0:$.getModel();B&&i&&((v=O.current)==null||v.editor.setModelLanguage(B,i))},[i],z),tt(()=>{var B;s!==void 0&&((B=L.current)==null||B.revealLine(s))},[s],z),tt(()=>{var B;(B=O.current)==null||B.editor.setTheme(o)},[o],z);let V=C.useCallback(()=>{var B;if(!(!F.current||!O.current)&&!ee.current){Y.current(O.current);let $=a||n,v=sr(O.current,r||e||"",t||i||"",$||"");L.current=(B=O.current)==null?void 0:B.editor.create(F.current,{model:v,automaticLayout:!0,...u},c),d&&L.current.restoreViewState(ca.get($)),O.current.editor.setTheme(o),s!==void 0&&L.current.revealLine(s),k(!0),ee.current=!0}},[e,t,n,r,i,a,u,c,d,o,s]);C.useEffect(()=>{z&&U.current(L.current,O.current)},[z]),C.useEffect(()=>{!E&&!z&&V()},[E,z,V]),j.current=r,C.useEffect(()=>{var B,$;z&&g&&((B=re.current)==null||B.dispose(),re.current=($=L.current)==null?void 0:$.onDidChangeModelContent(v=>{I.current||g(L.current.getValue(),v)}))},[z,g]),C.useEffect(()=>{if(z){let B=O.current.editor.onDidChangeMarkers($=>{var te;let v=(te=L.current.getModel())==null?void 0:te.uri;if(v&&$.find(we=>we.path===v.path)){let we=O.current.editor.getModelMarkers({resource:v});b==null||b(we)}});return()=>{B==null||B.dispose()}}return()=>{}},[z,b]);function y(){var B,$;(B=re.current)==null||B.dispose(),p?d&&ca.set(a,L.current.saveViewState()):($=L.current.getModel())==null||$.dispose(),L.current.dispose()}return jt.createElement(wh,{width:f,height:w,isEditorReady:z,loading:l,_ref:F,className:_,wrapperProps:S})}var Rb=Ab,Ib=C.memo(Rb),jb=Ib;function Ob({value:e,onChange:t,height:n="400px"}){const r=C.useCallback(i=>{t(i||"")},[t]);return x.jsx("div",{className:"h-full border border-gray-200 rounded-lg overflow-hidden shadow-sm",children:x.jsx(jb,{height:n,defaultLanguage:"python",theme:"light",value:e,onChange:r,options:{minimap:{enabled:!1},fontSize:14,lineNumbers:"on",scrollBeyondLastLine:!1,automaticLayout:!0,tabSize:4,wordWrap:"on",padding:{top:16,bottom:16},scrollbar:{verticalScrollbarSize:8,horizontalScrollbarSize:8}}})})}function Mb({output:e,isLoading:t=!1}){const n=C.useRef(null);return C.useEffect(()=>{var r;(r=n.current)==null||r.scrollIntoView({behavior:"smooth"})},[e]),x.jsxs("div",{className:"bg-white border border-gray-200 rounded-lg h-full flex flex-col shadow-sm",children:[x.jsxs("div",{className:"flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-gray-50",children:[x.jsx("span",{className:"text-sm font-medium text-gray-700",children:"Console Output"}),t&&x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-primary-500 rounded-full animate-pulse"}),x.jsx("span",{className:"text-xs text-gray-500",children:"Running..."})]})]}),x.jsxs("div",{className:"flex-1 overflow-auto p-4 font-mono text-sm bg-gray-900 rounded-b-lg",children:[e.length===0?x.jsx("div",{className:"text-gray-500 italic",children:"Output will appear here when you run your code..."}):e.map((r,i)=>x.jsx("div",{className:`py-0.5 ${r.startsWith("[Error]")?"text-red-400":r.includes("PASSED")?"text-green-400":r.includes("FAILED")?"text-red-400":"text-gray-300"}`,children:r.startsWith("Test")?x.jsxs("span",{children:[r.includes("PASSED")?"✓ ":r.includes("FAILED")?"✗ ":"",r]}):x.jsxs("span",{children:["> ",r]})},i)),x.jsx("div",{ref:n})]})]})}function Db({problem:e}){return x.jsxs("div",{className:"prose prose-sm max-w-none",children:[x.jsxs("div",{className:"flex items-center gap-3 mb-4",children:[x.jsx("h1",{className:"text-xl font-bold text-gray-900 m-0",children:e.title}),x.jsx("span",{className:`px-2 py-0.5 rounded text-xs font-medium ${e.difficulty==="easy"?"bg-green-100 text-green-700":e.difficulty==="medium"?"bg-yellow-100 text-yellow-700":"bg-red-100 text-red-700"}`,children:e.difficulty})]}),x.jsx(ch,{components:{h2:({children:t})=>x.jsx("h2",{className:"text-lg font-semibold text-gray-900 mt-6 mb-3",children:t}),h3:({children:t})=>x.jsx("h3",{className:"text-base font-medium text-gray-700 mt-4 mb-2",children:t}),p:({children:t})=>x.jsx("p",{className:"text-gray-600 mb-3 leading-relaxed",children:t}),code:({className:t,children:n})=>(t==null?void 0:t.includes("language-"))?x.jsx("pre",{className:"bg-gray-100 rounded-lg p-4 overflow-x-auto my-3",children:x.jsx("code",{className:"text-sm text-gray-800",children:n})}):x.jsx("code",{className:"bg-gray-100 px-1.5 py-0.5 rounded text-primary-600 text-sm",children:n}),ul:({children:t})=>x.jsx("ul",{className:"list-disc list-inside text-gray-600 space-y-1 mb-3",children:t}),li:({children:t})=>x.jsx("li",{className:"text-gray-600",children:t})},children:e.description})]})}function Fb({examples:e}){return x.jsxs("div",{className:"mt-6",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900 mb-3",children:"Examples"}),x.jsx("div",{className:"space-y-4",children:e.map((t,n)=>x.jsx("div",{className:"bg-white rounded-lg p-4 border border-gray-200 shadow-sm",children:x.jsxs("div",{className:"space-y-2",children:[x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Input:"}),x.jsx("pre",{className:"mt-1 text-sm text-primary-600 font-mono",children:t.input})]}),x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Output:"}),x.jsx("pre",{className:"mt-1 text-sm text-green-600 font-mono",children:t.output})]}),t.explanation&&x.jsxs("div",{className:"pt-2 border-t border-gray-200",children:[x.jsx("span",{className:"text-gray-500 text-sm",children:"Explanation: "}),x.jsx("span",{className:"text-gray-600 text-sm",children:t.explanation})]})]})},n))})]})}function Wb({hints:e,solution:t}){const[n,r]=C.useState(0),[i,a]=C.useState(!1),o=()=>{n<e.length&&r(l=>l+1)},s=()=>{a(!0)};return x.jsxs("div",{className:"mt-6 space-y-4",children:[x.jsxs("div",{children:[x.jsxs("div",{className:"flex items-center justify-between mb-3",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Hints"}),n<e.length&&x.jsxs("button",{onClick:o,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:["Show Hint (",n,"/",e.length,")"]})]}),n>0?x.jsx("div",{className:"space-y-2",children:e.slice(0,n).map((l,u)=>x.jsx("div",{className:"bg-yellow-50 border border-yellow-200 rounded-lg p-3",children:x.jsxs("div",{className:"flex items-start gap-2",children:[x.jsx("span",{className:"text-yellow-600",children:"💡"}),x.jsx("span",{className:"text-gray-700 text-sm",children:l})]})},u))}):x.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Show Hint" if you need help'})]}),x.jsxs("div",{children:[x.jsxs("div",{className:"flex items-center justify-between mb-3",children:[x.jsx("h3",{className:"text-base font-medium text-gray-900",children:"Solution"}),!i&&n>=e.length&&x.jsx("button",{onClick:s,className:"text-sm text-primary-600 hover:text-primary-500 transition-colors",children:"Reveal Solution"})]}),i?x.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[x.jsx("div",{className:"px-4 py-2 bg-gray-100 border-b border-gray-200",children:x.jsx("span",{className:"text-sm text-gray-600",children:"Solution Code"})}),x.jsx("pre",{className:"p-4 text-sm text-gray-300 font-mono overflow-x-auto bg-gray-900",children:t})]}):n<e.length?x.jsx("div",{className:"text-gray-400 text-sm italic",children:"Reveal all hints first to unlock the solution"}):x.jsx("div",{className:"text-gray-400 text-sm italic",children:'Click "Reveal Solution" to see the answer'})]})]})}function Bb({results:e,isRunning:t}){if(t)return x.jsxs("div",{className:"flex items-center gap-3 p-4 bg-white rounded-lg border border-gray-200 shadow-sm",children:[x.jsx("div",{className:"w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"}),x.jsx("span",{className:"text-gray-600",children:"Running tests..."})]});if(e.length===0)return null;const n=e.filter(i=>i.passed).length,r=e.filter(i=>!i.hidden);return x.jsxs("div",{className:"bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm",children:[x.jsxs("div",{className:"px-4 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50",children:[x.jsx("span",{className:"font-medium text-gray-900",children:"Test Results"}),x.jsxs("span",{className:`text-sm font-medium ${n===e.length?"text-green-600":"text-yellow-600"}`,children:[n,"/",e.length," passed"]})]}),x.jsxs("div",{className:"divide-y divide-gray-200",children:[r.map(i=>x.jsxs("div",{className:"p-4",children:[x.jsxs("div",{className:"flex items-center gap-2 mb-2",children:[x.jsx("span",{className:`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i.passed?"bg-green-100 text-green-600":"bg-red-100 text-red-600"}`,children:i.passed?"✓":"✗"}),x.jsxs("span",{className:"text-gray-700 text-sm font-medium",children:["Test ",i.id,": ",i.description]})]}),!i.passed&&x.jsxs("div",{className:"ml-7 space-y-1 text-sm",children:[x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500",children:"Expected: "}),x.jsx("span",{className:"text-green-600 font-mono",children:i.expected})]}),x.jsxs("div",{children:[x.jsx("span",{className:"text-gray-500",children:"Actual: "}),x.jsx("span",{className:"text-red-600 font-mono",children:i.actual})]})]})]},i.id)),e.some(i=>i.hidden)&&x.jsxs("div",{className:"p-4 text-gray-500 text-sm italic",children:["+ ",e.filter(i=>i.hidden).length," hidden test(s)",e.filter(i=>i.hidden&&i.passed).length===e.filter(i=>i.hidden).length?x.jsx("span",{className:"text-green-600 ml-2",children:"(all passed)"}):x.jsxs("span",{className:"text-red-600 ml-2",children:["(",e.filter(i=>i.hidden&&!i.passed).length," failed)"]})]})]})]})}function qb(){const{sectionId:e,problemId:t}=km(),{getProblemProgress:n,saveProblemCode:r,updateProblemStatus:i}=Mi(),{isLoading:a,isReady:o,output:s,runTests:l,clearOutput:u}=zk(),c=t?mk(t):void 0,d=Sr.find(N=>N.id===e),[p,f]=C.useState(""),[w,_]=C.useState([]),[S,h]=C.useState(!1);C.useEffect(()=>{if(c&&e){const N=n(e,c.id);f(N.code||c.starterCode)}},[c,e,n]),C.useEffect(()=>{if(p&&c&&e&&p!==c.starterCode){const N=setTimeout(()=>{r(e,c.id,p)},1e3);return()=>clearTimeout(N)}},[p,c,e,r]);const m=C.useCallback(N=>{f(N)},[]),g=C.useCallback(()=>{c&&(f(c.starterCode),_([]),u())},[c,u]),b=N=>{const O=N.match(/def\s+(\w+)\s*\(/);return O?O[1]:"solution"},z=C.useCallback(async()=>{if(!(!c||!o||!e)){h(!0),_([]);try{const N=b(c.starterCode),O=await l(p,c.testCases,N);_(O),O.every(F=>F.passed)&&i(e,c.id,"completed")}finally{h(!1)}}},[c,o,e,p,l,i]);if(!c||!d)return x.jsxs("div",{className:"text-center py-12",children:[x.jsx("h1",{className:"text-2xl font-bold text-gray-900 mb-4",children:"Problem Not Found"}),x.jsx(yn,{to:"/",className:"text-primary-600 hover:text-primary-500",children:"Return to Home"})]});const k=d.problems.indexOf(c.id),E=d.problems[k+1];return x.jsxs("div",{className:"h-[calc(100vh-8rem)] flex flex-col -m-6",children:[x.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-3",children:[x.jsxs(yn,{to:`/section/${e}`,className:"text-gray-500 hover:text-gray-900 transition-colors",children:["← ",d.title]}),x.jsx("span",{className:"text-gray-300",children:"/"}),x.jsx("span",{className:"text-gray-900 font-medium",children:c.title})]}),x.jsxs("div",{className:"flex items-center gap-3",children:[!o&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),a?"Loading Python...":"Python ready"]}),o&&x.jsxs("span",{className:"text-gray-500 text-sm flex items-center gap-2",children:[x.jsx("div",{className:"w-2 h-2 bg-green-500 rounded-full"}),"Python ready"]})]})]}),x.jsxs(to,{className:"flex flex-1 overflow-hidden",sizes:[40,60],minSize:300,gutterSize:8,gutterAlign:"center",children:[x.jsxs("div",{className:"overflow-auto p-6 bg-gray-50",children:[x.jsx(Db,{problem:c}),x.jsx(Fb,{examples:c.examples}),x.jsx(Wb,{hints:c.hints,solution:c.solution})]}),x.jsxs("div",{className:"flex flex-col overflow-hidden bg-white",children:[x.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200",children:[x.jsxs("div",{className:"flex items-center gap-2",children:[x.jsx("button",{onClick:z,disabled:!o||S,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:S?"Running...":"Run Tests"}),x.jsx("button",{onClick:g,className:"px-4 py-1.5 bg-gray-200 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-300 transition-colors",children:"Reset"})]}),w.length>0&&w.every(N=>N.passed)&&E&&x.jsx(yn,{to:`/problem/${e}/${E}`,className:"px-4 py-1.5 bg-green-100 text-green-700 text-sm font-medium rounded-md hover:bg-green-200 transition-colors",children:"Next Problem →"})]}),x.jsxs(to,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[x.jsx("div",{className:"h-full overflow-hidden p-4",children:x.jsx(Ob,{value:p,onChange:m,height:"100%"})}),x.jsxs("div",{className:"overflow-auto p-4 space-y-4 bg-gray-50",children:[x.jsx(Bb,{results:w,isRunning:S}),x.jsx(Mb,{output:s,isLoading:S})]})]})]})]})]})}function Ub(){return x.jsx(p0,{children:x.jsxs(Xx,{children:[x.jsx(ba,{path:"/",element:x.jsx(f0,{})}),x.jsx(ba,{path:"/section/:sectionId",element:x.jsx(gk,{})}),x.jsx(ba,{path:"/problem/:sectionId/:problemId",element:x.jsx(qb,{})})]})})}_s.createRoot(document.getElementById("root")).render(x.jsx(jt.StrictMode,{children:x.jsx(n0,{basename:"/ml-coding-lab",children:x.jsx(u0,{children:x.jsx(Ub,{})})})}));
